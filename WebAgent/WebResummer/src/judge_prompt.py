JUDGE_PROMPT_GAIA = """You are an evaluation assistant. Please determine if the predicted answer is equivalent to the labeled answer.

Question: {question}

Labeled Answer: {correct_answer}

Predicted Answer: {response}

Did the model give an answer **equivalent** to the labeled answer? Please respond with "Correct" if they are equivalent, or "Incorrect" if they are not equivalent. Do not include any other text.
"""


JUDGE_PROMPT_BC_zh = """
请根据给定问题、标准答案和模型预测的答案来评估模型的回答是否正确。您的任务是将结果评定为：【正确】、【错误】。

首先，我们将列出每个评定类别的示例，然后请您对新问题的预测答案进行评定。
以下是【正确】的答复示例：
```
问题：贝拉克·奥巴马的孩子叫什么名字？
标准答案：玛丽亚·奥巴马和萨莎·奥巴马
模型预测1：Malia Obama and Sasha Obama
模型预测2：玛丽亚和萨沙
模型预测3：大多数人会说是玛丽亚和萨莎，但我不确定，需要再确认
模型预测4：巴拉克·奥巴马有两个女儿，她们分别是玛丽亚·安和娜塔莎·玛丽安，但通常称作玛丽亚·奥巴马和萨莎·奥巴马。
```
这些答复均为【正确】，因为：
    - 完整地包含了标准答案中的重要信息。
    - 不包含任何与标准答案矛盾的信息。
    - 只关注语义内容，中英文，大小写、标点、语法和顺序不重要。
    - 答复中出现模糊语句或猜测是可以接受的，前提是包含了标准答案且不含有不正确信息或矛盾。

以下是【错误】的答复示例：
```
问题：巴拉克·奥巴马的孩子叫什么名字？
标准答案：玛丽亚·奥巴马和萨莎·奥巴马
模型预测1：玛丽亚
模型预测2：玛丽亚、萨莎和苏珊和萨莎·奥巴马或玛丽亚·奥巴马，或娜塔莎·玛丽安，或爱因斯坦
模型预测3：虽然我不知道他们的确切名字，但能说出巴拉克·奥巴马有两个孩子。
模型预测4：你可能是想说贝茜和奥利维亚。不过您应通过最新的参考资料确认详细信息。那是正确的答案吗？
模型预测5：巴拉克·奥巴马的孩子
```
这些答复均为【错误】，因为：
    - 答复中包含与标准答案矛盾的事实陈述。
    - 答案为空、重复表述问题。
    - 答案枚举了多个答案，重复表述答案。

需要格外注意的是：
- 标准答案中包含对于问题中多个方面的回答，并且在同一个方面的答案中可能会有多种不同的描述，这些描述均是正确的，并且在同一个括号中给出，通过逗号连接。例如，考虑问题"抖音自己的人工智能大模型叫什么名字？"，标准答案为"【【豆包，云雀】】"：
    - 预测答案"豆包"、"豆包、云雀"、"云雀"等均为【正确】。
- 对于标准答案中包含的不同方面的回答，模型需要同时给出所有方面的回答才可以算是正确，否则直接判断为【错误】，不存在【部分正确】这种输出方式，这些答案会在不同的括号中给出。例如，考虑问题"TFBOYS组合中的成员有哪些？"，标准答案为"【【王俊凯】【王源】【易洋千玺】】"：
    - 预测答案"王俊凯、王源、易洋千玺"等同时包含所有答案，才可以算为【正确】。
    - 预测答案为"王俊凯、易洋千玺"等没有同时包含所有答案，会被算为【错误】。

另外注意以下几点：
- 对于标准答案为数字的问题，预测答案应和标准答案一致。例如，考虑问题"金山铁路黄浦江特大桥的全长是多少米？"，标准答案为"3518.17"：
    - 预测答案"3518"、"3518.1"、"3518.17"均为【正确】。
    - 预测答案"3520"和"3600"均为【错误】。
- 如果模型预测并没有直接回答问题，模型试图绕过或未能直接给出标准答案视为【错误】答案。
    - 例如：问题"林宥嘉的老婆是谁"，标准答案为"丁文琪"。模型预测"林宥嘉的老婆"、"林宥嘉的老婆应该很优秀"、"林宥嘉的老婆可能是某个公众人物"均为【错误】。
- 如果标准答案包含比问题更多的信息，预测答案只需包含问题中提到的信息。
    - 例如，考虑问题"菱镁矿的主要化学成分是什么？"标准答案为"碳酸镁（MgCO3）"。"碳酸镁"或"MgCO3"均视为【正确】答案。
- 如果从问题中明显可以推断出预测答案省略的信息，那么算作正确。
    - 例如，问题"巴鲁米尼的努拉吉遗迹在1997年被联合国教科文组织列为世界文化遗产，那么这遗址在哪个地区？"标准答案为"意大利撒丁岛"，预测答案"撒丁岛"被视为【正确】。
- 如果能明显看出名字翻译版本不同但是是同一个人也认为正确。
    - 例如，如果标准答案是"Robinson"，那么回答鲁滨逊或者鲁滨孙均正确。
- 你应该更关注标准答案和模型预测的匹配度，而不是关心标准答案是否是正确的。

下面是一个新的问题示例。请只回复【正确】、【错误】之一，不要道歉或纠正自己的错误，只需要评估该回答。
```
问题: {question}
标准答案: {correct_answer}
预测答案: {response}
```

将此新问题的预测答案评定为以下之一：
A.【正确】
B.【错误】

只返回【正确】、【错误】所代表的选项即可，即仅返回A或B即可，无须添加任何其他的文本。
""".strip()


JUDGE_PROMPT_BC_en = """
Based on the given question, standard answer, and model-predicted answer, evaluate whether the model's response is correct. Your task is to classify the result as: [CORRECT] or [INCORRECT].

First, we'll list examples for each category, then you'll evaluate a new question's predicted answer.
Here are examples of [CORRECT] responses:
```
Question: What are the names of Barack Obama's children?
Standard Answer: Malia Obama and Sasha Obama
Model Prediction 1: Malia Obama and Sasha Obama
Model Prediction 2: Malia and Sasha
Model Prediction 3: Most would say Malia and Sasha, but I'm not sure, I should verify
Model Prediction 4: Barack Obama has two daughters, Malia Ann and Natasha Marian, commonly known as Malia Obama and Sasha Obama.
```
These responses are all [CORRECT] because they:
    - Fully include the important information from the standard answer.
    - Don't contain any information that contradicts the standard answer.
    - Focus only on semantic content; language, capitalization, punctuation, grammar, and order aren't important.
    - Vague statements or guesses are acceptable as long as they include the standard answer and don't contain incorrect information or contradictions.

Here are examples of [INCORRECT] responses:
```
Question: What are the names of Barack Obama's children?
Standard Answer: Malia Obama and Sasha Obama
Model Prediction 1: Malia
Model Prediction 2: Malia, Sasha and Susan or Sasha Obama or Malia Obama, or Natasha Marian, or Einstein
Model Prediction 3: While I don't know their exact names, I can tell you Barack Obama has two children.
Model Prediction 4: You might be thinking of Betsy and Olivia. But you should verify the details with the latest references. Is that the correct answer?
Model Prediction 5: Barack Obama's children
```
These responses are all [INCORRECT] because they:
    - Contain factual statements that contradict the standard answer.
    - Are empty or merely repeat the question.
    - Enumerate multiple answers or repeat the answer.

Pay special attention to the following:
- The standard answer may contain responses to multiple aspects of the question, and within the same aspect, there might be different descriptions, all of which are correct and are given in the same bracket, connected by commas. For example, for the question "What is the name of ByteDance's AI model?", the standard answer is "[[Doubao, Skylark]]":
    - Predicted answers "Doubao", "Doubao, Skylark", "Skylark", etc. are all [CORRECT].
- For standard answers containing responses to different aspects, the model needs to provide answers to all aspects to be considered correct; otherwise, it's directly judged as [INCORRECT]. There is no [PARTIALLY CORRECT] output option. These answers will be given in different brackets. For example, for the question "Who are the members of TFBOYS?", the standard answer is "[[Wang Junkai][Wang Yuan][Yi Yangqianxi]]":
    - Predicted answers like "Wang Junkai, Wang Yuan, Yi Yangqianxi" that include all answers are [CORRECT].
    - Predicted answers like "Wang Junkai, Yi Yangqianxi" that don't include all answers are [INCORRECT].

Also note the following points:
- For questions with numerical standard answers, the predicted answer should match the standard answer. For example, for the question "What is the total length in meters of the Huangpu River Bridge on the Jinshan Railway?", the standard answer is "3518.17":
    - Predicted answers "3518", "3518.1", "3518.17" are all [CORRECT].
    - Predicted answers "3520" and "3600" are [INCORRECT].
- If the model prediction doesn't directly answer the question, attempts to circumvent or fails to directly provide the standard answer, it's considered an [INCORRECT] answer.
    - For example, for the question "Who is JJ Lin's wife?", with the standard answer "Ding Wenqi", model predictions like "JJ Lin's wife", "JJ Lin's wife should be excellent", "JJ Lin's wife might be a public figure" are all [INCORRECT].
- If the standard answer contains more information than the question asks for, the predicted answer only needs to include the information mentioned in the question.
    - For example, for the question "What is the main chemical component of magnesite?", with the standard answer "Magnesium carbonate (MgCO3)", "Magnesium carbonate" or "MgCO3" are both considered [CORRECT] answers.
- If information omitted in the predicted answer can be clearly inferred from the question, it's considered correct.
    - For example, for the question "The Nuragic ruins of Barumini were listed as a World Cultural Heritage by UNESCO in 1997, so where is this site located?", with the standard answer "Sardinia, Italy", the predicted answer "Sardinia" is considered [CORRECT].
- If it's clear that different translations of a name refer to the same person, it's considered correct.
    - For example, if the standard answer is "Robinson", answers like "Lubinson" or "Lubinsun" are both correct.
- You should focus more on the match between the standard answer and the model prediction, rather than whether the standard answer itself is correct.

Below is a new question example. Please reply with only [CORRECT] or [INCORRECT], without apologies or corrections to your own errors, just evaluate the answer.
```
Question: {question}
Standard Answer: {correct_answer}
Predicted Answer: {response}
```

Evaluate this new question's predicted answer as one of the following:
A. [CORRECT]
B. [INCORRECT]

Return only the option representing [CORRECT] or [INCORRECT], i.e., just return A or B, without adding any other text.
""".strip()
