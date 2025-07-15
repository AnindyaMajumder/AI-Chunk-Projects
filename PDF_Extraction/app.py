from pdf_loader import load_pdf
from model import model_init

def main(claim_no: int, list_item: list[str], name: str, phone: str, email: str, pdf1: str = "pdf1.pdf", pdf2: str = "pdf2.pdf"):
    # Load and summarize policy document
    policy = load_pdf(pdf1)
    receipt = load_pdf(pdf2)

    # Generate comprehensive claim analysis
    claim_analysis_chain = model_init(claim_no, list_item, name, phone, email, policy, receipt)
    
    # Generate final claim analysis report
    final_response = claim_analysis_chain.invoke({
        "claim_no": claim_no,
        "name": name,
        "phone": phone,
        "email": email,
        "list_item": list_item,
        "policy": policy,
        "receipt": receipt
    })
    
    return final_response

if __name__ == "__main__":
    # Example usage
    claim_no = 123456
    list_item = ["Item1", "Item2"]
    name = "John Doe"
    phone = "123-456-7890"
    email = "johndoes@gmail.com"
    pdf1 = "pdf1.pdf"
    pdf2 = "pdf2.pdf"
    
    result = main(claim_no, list_item, name, phone, email, pdf1, pdf2)
    print(result)