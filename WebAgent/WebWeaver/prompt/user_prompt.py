

USER_PROMPT_INST = """A conversation between User and Assistant. The user asks a question, and the assistant follows the outline to write a long report or article by calling the tool and writing every section and paragraph.
The assistant should follow the sources provided in outline to retrieve related information for writing. If no sources are provided, the assistant should collect information from the <material> section.
The article should be as detailed as possible.
<tools>
{
  "name": "retrieve",
  description": "Read the webpage(s) whose id matches the given id and return the summary.",
  "parameters": {
  "type": "object",
  "properties": {
      "url_id": {
          "type": ["string", "array"],
          "items": {
              "type": "string"
              },
          "minItems": 1,
          "description": "The URL ID(s) of the webpage(s) to visit. Can be a single URL ID or an array of URL IDs."
    },
      "goal": {
            "type": "string",
            "description": "The goal of the visit for webpage(s)."
    }
  },
      "required": [
            "url_id", 
            "goal"
        ]
    }
}
</tools>
"""


USER_PROMPT_EXAMPLE = """
The assistant starts with one or more cycles of (thinking about which tool to use -> performing tool call -> waiting for tool response -> thinking what content can be utilized to answer the query -> write the section or paragraph), and ends with <terminate>. The thinking processes, tool calls, tool responses, writting content, and terminate signal are enclosed within their tags. There could be multiple thinking processes, tool calls, tool call parameters and tool response parameters.

Example response:
<think> thinking which tool is needed here </think>
<tool_call>
{"name": "tool name here", "arguments": {"parameter name here": parameter value here, "another parameter name here": another parameter value here, ...}}
</tool_call>
<tool_response>
tool_response here
</tool_response>
<think> thinking what content can be utilized to answer the query here </think>
<write> write the section or paragraph here </write>
<think> thinking which tool is needed here </think>
<tool_call>
{"name": "another tool name here", "arguments": {...}}
</tool_call>
<tool_response>
tool_response here
</tool_response>
<think> thinking what content can be utilized to answer the query here </think>
<write> write the section or paragraph here </write>
(more thinking processes, tool calls, tool responses and write here)
<think> thinking process here </think>
<terminate> the writing process is terminated.

User: """