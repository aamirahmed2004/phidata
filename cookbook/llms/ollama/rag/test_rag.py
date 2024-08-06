import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from app import query_llm
from assistant import get_rag_assistant
from langchain_community.llms.ollama import Ollama

EVAL_PROMPT = """
Expected Response: {expected_response}
Actual Response: {actual_response}
---
Answer with ONLY 'true' or 'false': Does the actual response match the expected response? 
"""

"""
    To run tests, use the following commands to output the results to a log file:

    $timestamp = Get-Date -Format "MM-dd_HH-mm"
    pytest . -v | tee "./logs/test_results_$timestamp.log"
"""

def test_monopoly_rules():
    # Monopoly: page 3
    question="How much total money does a player start with in Monopoly?"
    expected_response="$1500"
    assert query_and_validate(question, expected_response)

def test_SAM2():
    # SAM-2: page 18-19
    question="What was the distribution of the training data used to finetune SAM-2? Give me the sources and how much data was used from each source."
    expected_response="""The training data mixture consists
of ∼15.2% SA-1B, ∼70% SA-V and ∼14.8% Internal. The same settings are used when open-source datasets are included, with the change that the additional data is included (∼1.3% DAVIS, ∼9.4% MOSE, ∼9.2% YouTubeVOS, ∼15.5% SA-1B, ∼49.5% SA-V, ∼15.1% Internal)     
"""
    assert query_and_validate(question, expected_response)

def test_Triage_email_policy():
    # Triage: page 63
    question="What is TriageLogic's policy on email?"
    expected_response="""Email is not secure. TriageLogic requires that email cannot contain PHI anywhere in the heading or 
body of the message. To view any PHI, the user has to log into the system with a user name and 
password. When communicating about patients via email, we refer to ticket numbers or note numbers 
without mentioning any PHI. Reports and recordings are sent as links for which the user has to login to 
view the PHI. None of the following items can be included in any unsecure communication.  
1. Names 
2. All geographical identifiers smaller than a state, except for the initial three digits of a zip code 
if, according to the current publicly available data from the Bureau of the Census: the 
geographic unit formed by combining all zip codes with the same three initial digits contains 
more than 20,000 people; and the initial three digits of a zip code for all such geographic units 
containing 20,000 or fewer people is changed to 000 
3. Dates (other than year) directly related to an individual 
4. Phone numbers 
5. Fax numbers 
6. Email addresses 
7. Social Security numbers 
8. Medical record numbers 
9. Health insurance beneficiary numbers 
10. Account numbers 
11. Certificate/license numbers 
12. Vehicle identifiers and serial numbers, including license plate numbers; 
13. Device identifiers and serial numbers; 
14. Web Uniform Resource Locators (URLs) 
15. Internet Protocol (IP) address numbers 
16. Biometric identifiers, including finger, retinal and voice prints 
17. Full face photographic images and any comparable images 
18. Any other unique identifying number, characteristic, or code except the unique code assigned 
by the investigator to code the data
    """
    assert query_and_validate(question, expected_response)

def test_Triage_CORE39():
    # Triage: page 41-42
    question="Tell me TriageLogic's procedures from CORE39"
    expected_response="""We currently conduct surveys that fall into the following quality domains: proficiency, 
responsiveness, communication, accommodation, and patient satisfaction. The survey is also seen as a 
vehicle for early identification of problems and of actions required for continuous quality 
improvement. An average score of 3.8 or above overall satisfaction was used as the acceptable 
performance standard. The survey is performed randomly on a portion of patients that are called to 
take part in the survey  Patient survey is conducted at least once a year. 
• Proficiency is the customer’s perception of the capability, expertise, or knowledge of the staff 
and the manner in which services are provided. 
• Responsiveness includes timeliness, assistance, and guidance. 
• Communication focuses on clarity of verbal and written expression. 
• Accommodation regards the behavior or interpersonal skills of staff; statements in this domain 
focus on respect, courtesy and sensitivity. 
• Patient Satisfaction pertains to the overall patient impression of the encounter with staff; 
statements in this domain focus on improving services offered to clients and the appropriateness 
of the surveillance process. 
Results: 
The results of the client and consumer surveys will be tabulated and reported to the CEO and Quality 
Committee. The committee will review all survey results and determine priorities for improvement. 
All specific areas of concern will be incorporated into the Quality Improvement Program.
"""
    assert query_and_validate(question, expected_response)

def test_Nomad_amount_and_years():
    # Nomad: page 3
    question="How much did the Nomad Investment fund earn its clients over its duration and how long did they operate?"
    expected_response="""The Nomad Investment Fund earned its clients $2 billion over its duration. It operated for 13 years, from 2001 to 2014."""
    assert query_and_validate(question, expected_response)

def test_Nomad_initial_investment():
    # Nomad: page 5-6
    question="Tell me about Nomad Investment fund's initial investments and what they were looking for in their investments."
    expected_response="""According to the provided information, Nomad Investment Partnership
was launched on September 10th, 2001, and began investing shortly
after. As of December 2001, the fund had made 18 investments across 16
different sectors, as classified by Bloomberg.

The initial investments were concentrated in certain sectors, with
media (TV, newspapers, and publishing) making up around 21% of the
portfolio, followed by hotels, resorts, and casinos at 12%, and
telecom services (mobile and cable) at 10%. Geographically, the
majority of the fund's investments were in South East Asia (32.8%),
North America (23.6%), Europe (12.7%), and other emerging markets
(2.9%).

When evaluating potential investments, Nomad was looking for
businesses trading at around half of their real business value,
companies run by owner-oriented management teams, and employing
capital allocation strategies consistent with long-term shareholder
wealth creation. This approach allowed Nomad to focus on finding
undervalued firms that would generate strong returns over the long
term.

In this context, two specific investments highlighted in the inaugural
annual letter were International Speedway (US) and Matichon
(Thailand). These examples illustrate Nomad's investment strategy and
its focus on finding undervalued opportunities with potential for
significant returns.
"""
    assert query_and_validate(question, expected_response)

# def test_Nomad_comparison():
#     assert query_and_validate(
#         question = "Compare the Nomad Investment fund and the World Index for the first six years of the Nomad fund's operations.",
#         expected_response="""

# """
#     )

def test_Buffet_1963():
    # Buffet: page 51
    question="In detail, what does Buffet say about his fund's performance in 1963?"
    expected_response="""1963 was a good year. It was not a good year because we had an overall gain of $3,637,167 or 38.7% on our 
beginning net assets, pleasant as that experience may be to the pragmatists in our group. Rather it was a good 
year because our performance was substantially better than that of our fundamental yardstick --the Dow-Jones 
Industrial Average (hereinafter called the “Dow”). If we had been down 20% and the Dow had been down 30%, 
this letter would still have begun “1963 was a good year.” Regardless of whether we are plus or minus in a 
particular year, if we can maintain a satisfactory edge on the Dow over an extended period of time, our long 
term results will be satisfactory -- financially as well as philosophically.
"""
    assert query_and_validate(question, expected_response)


def test_Buffet_table_comparison():
    # Buffet: page 51 but also others 
    question="Compare the performance of the Buffet Partnership to the Dow Jones Industrial Average from 1958 to 1962. Give me the percentage returns for each year."
    expected_response="""1958 - Buffett Partnership: 40.9%, Dow Jones Industrial Average: 38.5%
    1959 - Buffett Partnership: 25.9%, Dow Jones Industrial Average: 20.0%
    1960 - Buffett Partnership: 22.8%, Dow Jones Industrial Average: -6.2%
    1961 - Buffett Partnership: 45.9%, Dow Jones Industrial Average: 22.4%
    1962 - Buffett Partnership: 13.9%, Dow Jones Industrial Average: -7.6%
"""
    assert query_and_validate(question, expected_response)

def test_Buffet_fetch_value_from_table():
    # Buffet: page 54
    question="How much would $100k compounding at 12% per year make over 20 years?"
    expected_response="$864,627"
    assert query_and_validate(question, expected_response)

def test_Buffet_summarize():
    # Buffet: page 26-29
    question="Give me a detailed summary of Buffet's letter from 6th July, 1962."
    expected_response="""Prediction and Strategy Overview:

Buffett emphasizes long-term perspective: Predicting that the Dow Jones Industrial Average (DJIA) will likely yield 5% to 7% annual returns over a long period, he stresses that expecting much higher returns may lead to disappointment.
Relative Performance Focus: The goal of the partnership is to outperform the DJIA by an average of 10 percentage points per year. Buffett prefers years where the partnership outperforms the market, even if absolute returns are negative, as opposed to simply matching positive market returns.
First Half of 1962 Performance:

Market Decline and Partnership Performance: The DJIA fell 21.7% in the first half of 1962, while the partnership declined only 7.5%, showcasing its conservative approach.
Emphasis on Down Markets: Buffett highlights that the partnership's strategy is particularly effective in declining or static markets, providing a significant advantage during such periods.
Comparison with Investment Companies:

Investment Companies Underperformance: The letter compares the partnership's performance with large mutual and closed-end funds, showing that the partnership consistently outperforms these professional managers, who typically achieve results similar to the DJIA.
Future Expectations and Cautions:

No Guarantees: Buffett acknowledges that predictions, especially regarding market movements, are inherently uncertain and emphasizes that short-term results can fluctuate widely.
Partnership's Conservative Nature: He reiterates the partnership's conservative investment approach, aiming for less downside in bad years and acceptable performance in good years, to achieve satisfactory long-term results.
Asset Values and Payment Adjustments:

Impact on Partners: For partners receiving monthly payments, any reduction in market value equity will result in lower payments in subsequent years. Buffett clarifies the impact of potential losses on partner equity and future distributions.
"""
    assert query_and_validate(question, expected_response)

def test_Buffet_complex_table():
    # Buffet: page 44
    question = "From Berkshire's 1961 balance sheets, give me their adjusted valuation of current assets."
    expected_response = "$3,593,000"
    assert query_and_validate(question, expected_response)

def test_Buffet_Commonwealth():
    # Buffet: page 4
    question = "What percentage of the assets of Buffet's various partnerships did the Commonwealth Trust Co. represent?"
    expected_response = "Approximately 10% - 20%"
    assert query_and_validate(question, expected_response)

def test_Buffet_Commonwealth():
    # Buffet: page 4
    question = "What was the intrinsic value per share of Commonwealth Trust Co. when Buffett started purchasing the stock? "
    expected_response = "Around $125 per share."
    assert query_and_validate(question, expected_response)

def test_Buffet_Sanborn():
    # Buffet: page 12
    question = "How many stockholders had shares of Sanborn?"
    expected_response = "There were 1600 stockholders before half of them exchanged their stock for portfolio securities at fair value."
    assert query_and_validate(question, expected_response)

def query_and_validate(question: str, expected_response: str, llm: str = "llama3", embeddings_model: str = "nomic-embed-text"):

    rag_assistant = get_rag_assistant(llm_model=llm, embeddings_model=embeddings_model)
    response_text = query_llm(rag_assistant, question)

    prompt = EVAL_PROMPT.format(
        expected_response=expected_response, actual_response=response_text
    )

    model = Ollama(model="llama3")
    evaluation_results_str = model.invoke(prompt)
    evaluation_results_str_cleaned = evaluation_results_str.strip().lower()

    print(prompt)

    if "true" in evaluation_results_str_cleaned:
        # Print response in Green if it is correct.
        print("\033[92m" + f"Response: {evaluation_results_str_cleaned}" + "\033[0m")
        return True
    elif "false" in evaluation_results_str_cleaned:
        # Print response in Red if it is incorrect.
        print("\033[91m" + f"Response: {evaluation_results_str_cleaned}" + "\033[0m")
        return False
    else:
        raise ValueError(
            f"Invalid evaluation result. Cannot determine if 'true' or 'false'."
        )
