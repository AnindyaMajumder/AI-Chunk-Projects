from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

def model_init(claim_no: int, list_item: list[str], name: str, phone: str, email: str, policy: str, receipt: str):
    load_dotenv()
    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0.3,
        max_tokens=2048,
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )

    template = ChatPromptTemplate.from_messages([
        ("system", """You are an expert insurance claim processor and document analyzer. Your role is to generate comprehensive claim analysis reports by consolidating information from multiple sources. You must preserve ALL numerical data, dates, amounts, policy numbers, and specific details without any approximation or loss of precision."""),
        ("user", """Generate a comprehensive insurance claim analysis report using the provided information:

        CLAIM DETAILS:
        - Claim Number: {claim_no}
        - Claimant Name: {name}
        - Contact Phone: {phone}
        - Contact Email: {email}
        - Claimed Items: {list_item}

        POLICY DOCUMENT:
        {policy}

        RECEIPT/DOCUMENTATION:
        {receipt}

        ANALYSIS REQUIREMENTS:
        1. CLAIM VALIDATION: Cross-reference the claimed items against policy coverage and provided receipts
        2. NUMERICAL ACCURACY: Preserve ALL exact amounts, dates, policy numbers, and quantities
        3. COVERAGE ASSESSMENT: Determine coverage eligibility for each claimed item
        4. DISCREPANCY IDENTIFICATION: Highlight any inconsistencies between policy, receipts, and claimed items
        5. FINANCIAL SUMMARY: Calculate total claim amount, covered amounts, deductibles, and out-of-pocket costs
        6. RECOMMENDATIONS: Provide approval/denial recommendations with detailed justifications

        OUTPUT FORMAT:
        Structure your response with clear sections including:
        - Executive Summary
        - Claim Details Verification
        - Item-by-Item Analysis
        - Financial Breakdown (with exact figures)
        - Policy Compliance Assessment
        - Final Recommendations
        - Required Next Steps

        CRITICAL: Maintain complete accuracy of all numbers, dates, names, and financial data. Do not round, approximate, or omit any numerical information.""")])
    
    chain = template | llm | StrOutputParser()
    
    return chain