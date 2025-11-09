import ast
import io
import tokenize

def check_banned_operations(code: str):
    """
    Scans Python code for certain banned keywords (exit, yield, requests, url,
    pip, install, conda) as real code tokens, ignoring strings/comments
    to reduce false positives.
    
    Returns:
        (True, "No banned operations detected") if safe,
        (False, "Banned Error message") if a banned token is found.
    """

    # Define sets of banned tokens by category:
    banned_system_ops = {"exit", "yield"}
    banned_web_ops = {"requests", "url"}
    banned_install_ops = {"pip", "install", "conda"}

    # Wrap the code into a stream for the tokenize module
    code_stream = io.StringIO(code)

    try:
        # Generate tokens
        for token in tokenize.generate_tokens(code_stream.readline):
            # token is a namedtuple of the form:
            # TokenInfo(type, string, start, end, line)

            token_type = token.type
            token_str = token.string

            # Skip comments and strings: these won't trigger a ban
            if token_type in (tokenize.COMMENT, tokenize.STRING):
                continue

            # If it's a keyword (like 'yield'), Python's tokenizer sets type=NAME
            # for 'exit' (which is not a built-in keyword) but type=NAME or type=NAME/KEYWORD
            # for 'yield' depending on the Python version. We'll unify by checking
            # the token.string if it's exactly in our sets.

            # Check system ops
            if token_str in banned_system_ops:
                return (False, f"Banned Error: Using system operation '{token_str}' is not allowed.")
            # Check web ops
            if token_str in banned_web_ops:
                return (False, f"Banned Error: Using web operation '{token_str}' is not allowed.")
            # Check install ops
            if token_str in banned_install_ops:
                return (False, f"Banned Error: Using installation operation '{token_str}' is not allowed.")

    except tokenize.TokenError as e:
        # If the code is malformed, you could treat it as an error or ignore it
        return (False, f"Tokenization Error: {e}")

    # If we reach here, we found no banned tokens
    return (True, "No banned operations detected")


class CodeSafetyChecker:
    """
    A class to analyze Python code (as a string) and detect potentially dangerous
    file/system/subprocess operations. It uses an internal AST NodeVisitor to
    check imports and function calls.
    """

    # Known dangerous functions: (module_name, function_name)
    dangerous_functions = {
        ('os', 'remove'), ('os', 'unlink'), ('os', 'rmdir'), ('os', 'removedirs'),
        ('os', 'rename'), ('os', 'renames'), ('os', 'chmod'), ('os', 'chown'),
        ('os', 'system'), ('os', 'spawnl'), ('os', 'spawnle'), ('os', 'spawnlp'),
        ('os', 'spawnlpe'), ('os', 'spawnv'), ('os', 'spawnve'), ('os', 'spawnvp'),
        ('os', 'spawnvpe'), ('shutil', 'rmtree'), ('shutil', 'move'), ('shutil', 'copy'),
        ('shutil', 'copytree'), ('subprocess', 'run'), ('subprocess', 'call'),
        ('subprocess', 'check_call'), ('subprocess', 'check_output'), ('subprocess', 'Popen'),
        ('builtins', 'open'), ('builtins', 'eval'), ('builtins', 'exec')
    }

    # Write/append modes we consider dangerous (or at least worth flagging).
    # We allow reading modes ("r", "rb", etc.) or no mode argument (default read).
    dangerous_open_modes = {
        'w', 'a', 'x', 'w+', 'a+', 'x+', 'wb', 'ab', 'xb', 'w+b', 'a+b', 'x+b'
        # Optionally consider "r+" or "rb+" or "rt+" if you treat them as read-write
        'r+', 'rb+', 'rt+'
    }

    # Methods that typically write data to disk, e.g., DataFrame.to_csv(...)
    # You can expand this list to "to_excel", "to_json", "to_parquet", etc. if desired.
    dangerous_write_methods = {
        'to_csv',
        'to_excel',
        'to_json',
        'to_parquet',
        'to_pickle',
        # etc...
    }

    class _DangerousCodeVisitor(ast.NodeVisitor):
        """
        A specialized AST NodeVisitor that inspects import statements and function
        calls to detect if the code calls any known dangerous operations.
        """
        def __init__(self, parent_checker):
            super().__init__()
            self.parent = parent_checker
            # Maps local import name -> (module_name, original_function_name or None)
            # e.g. after `import os as foo`, we get: {'foo': ('os', None)}
            # or after `from os import remove as rm`, we get: {'rm': ('os', 'remove')}
            self.import_map = {}
            self.dangerous_operations = []

        def visit_Import(self, node):
            """
            Visits 'import X' or 'import X as Y' nodes and stores them in import_map.
            """
            for alias in node.names:
                module = alias.name
                alias_name = alias.asname or module
                # We store (actual_module_name, None) to indicate a normal import
                self.import_map[alias_name] = (module, None)
            self.generic_visit(node)

        def visit_ImportFrom(self, node):
            """
            Visits 'from X import Y' or 'from X import Y as Z' nodes and stores them.
            If the import is from a known "dangerous" module (os, shutil, subprocess),
            we also check for wildcard imports and mark them as potentially dangerous.
            """
            module = node.module
            for alias in node.names:
                if alias.name == '*':
                    # Wildcard import from these modules is flagged because
                    # it’s unclear what exactly is being imported
                    if module in {'os', 'shutil', 'subprocess'}:
                        self.dangerous_operations.append(
                            f"Wildcard import from '{module}' module"
                        )
                    continue
                original_name = alias.name
                alias_name = alias.asname or original_name
                self.import_map[alias_name] = (module, original_name)
            self.generic_visit(node)

        def visit_Call(self, node):
            """
            Visits all function calls. Depending on whether the function is accessed
            via a simple Name (e.g. open(...)) or an Attribute (e.g. os.remove(...)),
            checks against known dangerous functions or modules.
            """
            if isinstance(node.func, ast.Name):
                # e.g. open(...), remove(... if from "from os import remove")
                func_name = node.func.id
                self._handle_func_name_call(func_name, node)
            elif isinstance(node.func, ast.Attribute):
                # e.g. os.remove(...), shutil.copy(...)
                self._handle_attribute_call(node)
            self.generic_visit(node)

        def _handle_func_name_call(self, func_name, node):
            """
            If the function is simply called by name (e.g. open, remove, etc.),
            figure out which module it's from (if any) and check if it's dangerous.
            """
            if func_name in self.import_map:
                # e.g. 'remove' mapped to ('os', 'remove') or 'os', None
                module, original_func = self.import_map[func_name]
                real_func = original_func if original_func is not None else func_name
                if (module, real_func) in self.parent.dangerous_functions:
                    self._report_dangerous_call(module, real_func, node)
            else:
                # Possibly a builtin like open, eval, or exec
                if func_name in ('open', 'eval', 'exec'):
                    self._check_builtin(func_name, node)

        def _handle_attribute_call(self, node):
            """
            Handle calls of the form object.method(...) or module_name.function_name(...).
            """
            attr_name = node.func.attr

            # 1) If it's an explicit module call like: os.remove(...), or df.to_csv(...)
            if isinstance(node.func.value, ast.Name):
                module_or_var_name = node.func.value.id

                # Check if it's a known dangerous module method
                if module_or_var_name in self.import_map:
                    real_module, original_func = self.import_map[module_or_var_name]
                    # e.g. if we did "import os", then real_module="os", original_func=None
                    if (real_module, attr_name) in self.parent.dangerous_functions:
                        self._report_dangerous_call(real_module, attr_name, node)
                    elif real_module == 'subprocess' and attr_name in {
                        'run', 'call', 'check_call', 'check_output', 'Popen'
                    }:
                        self._check_subprocess_shell(node, attr_name)
                else:
                    # e.g. "os.remove(...)" if we did "import os", or "df.to_csv(...)"
                    if (module_or_var_name, attr_name) in self.parent.dangerous_functions:
                        # e.g. (os, remove) in the set
                        self._report_dangerous_call(module_or_var_name, attr_name, node)
                    elif module_or_var_name == 'subprocess' and attr_name in {
                        'run', 'call', 'check_call', 'check_output', 'Popen'
                    }:
                        self._check_subprocess_shell(node, attr_name)

                # 2) If it's a known “write” method like `df.to_csv()`,
                #    we treat it as dangerous. We do not care about the variable name
                #    here; *any* object calling `.to_csv(...)` is flagged.
                if attr_name in self.parent.dangerous_write_methods:
                    self._report_method_write(attr_name)
            else:
                # If node.func.value is another AST node type (like complex expression),
                # it's more advanced to detect. We'll handle the "attr_name" check anyway:
                if attr_name in self.parent.dangerous_write_methods:
                    self._report_method_write(attr_name)

        def _report_dangerous_call(self, module, func_name, node):
            """
            Handle recognized "dangerous" calls, with special logic for open(...).
            """
            if (module, func_name) == ('builtins', 'open') or module == 'open':
                mode = self._get_open_mode(node)
                if mode in self.parent.dangerous_open_modes:
                    self.dangerous_operations.append(f"Call to 'open' with mode '{mode}'")
                # If mode is None or read-only, it's allowed
            else:
                # Generic catch for anything else in the dangerous list
                self.dangerous_operations.append(f"Call to '{module}.{func_name}'")

        def _report_method_write(self, method_name: str):
            """
            We discovered something like <object>.to_csv(...),
            so we mark it as potentially dangerous since it writes to disk.
            """
            self.dangerous_operations.append(f"Call to method '{method_name}' (writes to file)")

        def _check_builtin(self, func_name, node):
            """
            Direct call to builtins like open, eval, exec with no import.
            """
            if func_name == 'open':
                mode = self._get_open_mode(node)
                if mode in self.parent.dangerous_open_modes:
                    self.dangerous_operations.append(f"Call to 'open' with mode '{mode}'")
            else:
                # eval or exec
                self.dangerous_operations.append(f"Call to built-in '{func_name}'")

        def _check_subprocess_shell(self, node, func_name):
            """
            If we see subprocess.run(..., shell=True), or similar calls,
            flag it as especially dangerous.
            """
            for kw in node.keywords:
                if kw.arg == 'shell' and isinstance(kw.value, ast.Constant) and kw.value.value is True:
                    self.dangerous_operations.append(
                        f"Call to 'subprocess.{func_name}' with 'shell=True'"
                    )

        def _get_open_mode(self, node):
            """
            Extract the 'mode' argument if the user calls open(..., mode='w'), etc.
            """
            mode = None
            # Check positional arguments
            if len(node.args) >= 2:
                mode_arg = node.args[1]
                if isinstance(mode_arg, ast.Constant) and isinstance(mode_arg.value, str):
                    mode = mode_arg.value

            # Check keyword arguments
            for kw in node.keywords:
                if kw.arg == 'mode' and isinstance(kw.value, ast.Constant):
                    if isinstance(kw.value.value, str):
                        mode = kw.value.value

            return mode

    def check_code_safety(self, tree):
        """
        Parse the given Python AST node, and detect known
        dangerous function calls or file operations. Returns (is_safe, message).
        """
        visitor = self._DangerousCodeVisitor(self)
        visitor.visit(tree)

        if visitor.dangerous_operations:
            message = (
                "Safety Error: This is not allowed, please do not perform dangerous or expensive file operations.\n- "
                + "\n- ".join(visitor.dangerous_operations)
            )
            return (False, message)
        else:
            return (True, "No dangerous file operations detected.")


if __name__ == "__main__":
    dangerous_code1 = r'''
import os

def my_func():
    os.remove("debug_1234.jsonl")
'''
    dangerous_code2 = r'''
import subprocess

subprocess.run(["rm", "-rf", "debug_1234/"])
'''
    dangerous_code3 = r'''
with open("debug_1234.csv", "w") as f:
    f.write("writing attempt")
'''
    dangerous_code4 = r'''
import pandas as pd

df = pd.read_csv('debug_1234.csv')
df.to_csv('debug_1234.csv', index=False)
'''

    safe_code = r'''
import pandas as pd

df = pd.read_csv('debug_1234.csv')
print(df)
with open('debug_1234.jsonl', 'r') as f:
    for line in f:
        print(line)
'''

    custom_code = r'''

'''

    safe, message = check_banned_operations(custom_code)
    print("Safe code?", safe)
    print("Message:", message)

    checker = CodeSafetyChecker()
    try:
        tree = ast.parse(custom_code)
    except SyntaxError as e:
        raise SyntaxError(f"Syntax Error: {e}")

    safe, message = checker.check_code_safety(tree)
    print("Safe code?", safe)
    print("Message:", message)
