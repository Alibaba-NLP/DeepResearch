from markdown_it import MarkdownIt
import html
import re

def get_style_css(style_name):
    """
    根据选择的样式名称获取对应的CSS样式文件
    
    Args:
        style_name (str): 样式名称，可选值为"Default"、"MBE"、"Glassmorphism"、"Apple"
    
    Returns:
        str: CSS样式内容
    """
    
    if style_name == "Default":
        return open("assets/demo.css", "r").read()
    elif style_name == "1":
        return open("assets/demo.1.css", "r").read()
    elif style_name == "MBE":
        return open("assets/demo_mbe.css", "r").read()
    elif style_name == "Glassmorphism":
        return open("assets/demo_glassmorphism.css", "r").read()
    elif style_name == "Apple":
        return open("assets/demo_apple.css", "r").read()
    elif style_name == "Paper":
        return open("assets/demo_paper.css", "r").read()
    else:
        return open("assets/demo.css", "r").read()

def decorate_writing(writing_result, style="Default"):
    if not writing_result:
        return writing_result
    
    cite_pattern = r'<qwen:cite\s+url=["\']([^"\']+)["\'](?:\s+[^>]*)?>(.*?)</qwen:cite>'
    takeaway_pattern = r'<qwen:takeaway(?:\s+class=["\'](?P<class>[^"\']+)["\'])?>(?P<content>[^<]*)</qwen:takeaway>'
    citation_map = {}

    def replace_cite(match):
        nonlocal citation_map
        urls = match.group(1).split(',')
        content = match.group(2)
        citation_html = []
        
        for url in urls:
            if url not in citation_map:
                citation_map[url] = len(citation_map) + 1
            current_index = citation_map[url]
            citation_html.append((f'<a href="{url}" title="点击查看引用来源: {url}">{current_index}</a>', current_index))
        
        citation_html = sorted(citation_html, key=lambda x: x[1])
        citation_html = ', '.join([x[0] for x in citation_html])

        cite_html = f'{content}<sup class="citation">[{citation_html}]</sup>'
        return cite_html
    
    decorated_result = re.sub(cite_pattern, replace_cite, writing_result, flags=re.S)

    def replace_takeaway(match):
        class_attr = match.group('class')
        content = match.group('content')
        
        if class_attr:
            return f'<div class="takeaway {class_attr}">{content}</div>'
        else:
            return f'<div class="takeaway">{content}</div>'
    
    decorated_result = re.sub(takeaway_pattern, replace_takeaway, decorated_result, flags=re.S)

    mermaid_pattern = r'```mermaid\n(.*?)\n```'
    def decorate_mermaid(match):
        return f"""
<pre class="mermaid">
{match.group(1)}
</pre>
"""
    decorated_result = re.sub(mermaid_pattern, decorate_mermaid, decorated_result, flags=re.S)

    echarts_pattern = r'```echarts\n(.*?)\n```'
    echarts_index = 0
    
    def replace_echarts(match):
        """
        将echarts代码块转换为HTML和JavaScript
        
        Args:
            match: 正则表达式匹配对象
        
        Returns:
            str: 包含HTML和JavaScript的echarts图表代码
        """
        nonlocal echarts_index 
        echarts_code = match.group(1)
        echarts_id = f'echarts-container-{echarts_index}'
        echarts_index += 1
        
        replace_code = f"""
<div class="echarts-container loading" id="{echarts_id}">Echarts Rendering...</div>
<script>
    var chartDom = document.getElementById('{echarts_id}');
    var myChart = echarts.init(chartDom);
    var option;
    option = {echarts_code};
    myChart.setOption(option);
    chartDom.classList.remove('loading');
</script>
        """
        return replace_code
    
    decorated_result = re.sub(echarts_pattern, replace_echarts, decorated_result, flags=re.S)

    md = MarkdownIt()
    body = md.render(decorated_result)
    
    selected_css = get_style_css(style)
    
    html_content = """
<html>
<head>
    <!-- KaTeX for mathematical formulas -->
    <link rel="stylesheet" href="https://s4.zstatic.net/npm/katex@0.16.0/dist/katex.min.css">
    <script src="https://s4.zstatic.net/npm/katex@0.16.0/dist/katex.min.js"></script>
    <script src="https://s4.zstatic.net/npm/katex@0.16.0/dist/contrib/auto-render.min.js"></script>
    <script src="https://s4.zstatic.net/npm/echarts@5.6.0/dist/echarts.min.js"></script>
    <style>
""" + selected_css + """
    </style>
</head>
<body>
<div class="generated-content">
""" + body + """</div>
<script type="module">
    import mermaid from 'https://unpkg.com/mermaid@11.6.0/dist/mermaid.esm.min.mjs';
</script>
<script>
    document.addEventListener('DOMContentLoaded', function() {
        renderMathInElement(document.body);
    });
</script>
</body>
</html>
"""
    # 转义HTML内容以便在iframe中安全使用
    # 这是必要的，因为HTML内容包含引号和其他特殊字符
    escaped_html_content = html.escape(html_content)
    
    # 定义iframe的样式属性
    iframe_style = "width: 100%; height: 1024px; transform-origin: top left; border-color: lightgrey; border-width: 1px; border-radius: 10px;"
    
    # 创建最终的iframe HTML，通过srcdoc属性注入转义后的HTML内容
    # 设置loading="eager"和importance="high"以优先加载
    # pointer-events="none"防止用户与iframe内容交互
    iframe_content = f'<iframe id="ai-ui-iframe" loading="eager" importance="high" pointer-events="none" style="{iframe_style}" srcdoc="{escaped_html_content}"></iframe>'

    # 返回最终的iframe HTML内容
    iframe_content = re.sub(r'\n\s*\n', '\n', iframe_content)
    return iframe_content
