# Topics as Proxies for Sociodemographics: How Conversational Context Affects LLM Answers
This is the repository for *Topics as Proxies for Sociodemographics: How Conversational Context Affects LLM Answers.*

![Introduction](figure.png)

## Paper abstract
When large language models (LLMs) are used in high-stakes scenarios, such as legal, medical and financial advice, even a single conversation history is enough to drive differences in outcomes between users. Prior work has demonstrated that this results in outcome disparities between sociodemographic groups, with some groups receiving more advantageous outcomes than others. In this work, we demonstrate that LLMs actually struggle to infer user sociodemographics from a single conversation history and that although there are disparities between sociodemographic groups, they are minimal in magnitude. To investigate what is the main driver of these disparities, we compare user sociodemographics to a range of (psycho)linguistic features of conversations, including conversation topic, emotions and readability. We find that conversation topics are most predictive of LLM-generated advice in conversational context, functioning as proxies for sociodemographic groups and often affecting advice in unpredictable ways. Since the resulting biases are subtle, this is cause for concern and highlights the need for future research to determine how best to detect and mitigate them when they are undesirable.

## Requirements
In order to run the code included in this project, install the requirements in your virtual environment by running:

```
pip install -r requirements.txt
```
This project was developed using Python 3.12.

## Using this repository
Please download:
- prompts from https://github.com/MatthewTKearney/sociolinguistic-bias-benchmark/tree/main
- concreteness scores from https://link.springer.com/article/10.3758/s13428-013-0403-5#MOESM1
- topic cluster information for PRISM from https://github.com/HannahKirk/prism-alignment

and place all in the `data` folder before continuing.
