import matplotlib.pyplot as plt

def create_pdf(filename, text):
    fig = plt.figure(figsize=(8, 11))
    fig.text(0.1, 0.9, text, fontsize=12, wrap=True)
    fig.savefig(filename, format='pdf')
    plt.close(fig)

text_content = """What is AWS IAM?
AWS Identity and Access Management (IAM) is a web service that helps you securely control access to AWS resources. You use IAM to control who is authenticated (signed in) and authorized (has permissions) to use resources.

When you first create an AWS account, you begin with a single sign-in identity that has complete access to all AWS services and resources in the account. This identity is called the AWS account root user and is accessed by signing in with the email address and password that you used to create the account.

IAM Features:
- Shared access to your AWS account
- Granular permissions
- Secure access to AWS resources for applications that run on Amazon EC2
- Multi-factor authentication (MFA)
- Identity federation
- Identity information for assurance
- PCI DSS Compliance
- Eventually Consistent
"""

create_pdf("data/raw_docs/aws_iam.pdf", text_content)
print("Created dummy PDF: data/raw_docs/aws_iam.pdf")
