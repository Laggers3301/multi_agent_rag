import os
import glob

try:
    import PyPDF2
except ImportError:
    os.system("pip install PyPDF2")
    import PyPDF2

pdf_dir = r"c:\Users\18207\Desktop\UCAS\Intern\multi_agent_rag\paper"
pdf_files = glob.glob(os.path.join(pdf_dir, "*.pdf"))

for pdf_file in pdf_files:
    print(f"--- Extraction from {os.path.basename(pdf_file)} ---")
    try:
        with open(pdf_file, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            text = ""
            # Extract first 3 pages to find benchmarks mentioned in abstract/introduction/experiments
            for i in range(min(3, len(reader.pages))):
                text += reader.pages[i].extract_text()

            # Simple keyword search for common benchmarks
            benchmarks = [
                "GAIA",
                "SWE-bench",
                "HotpotQA",
                "WebArena",
                "HumanEval",
                "AgentBench",
                "ALFWorld",
                "WebShop",
                "MMLU",
                "DROP",
                "MATH",
            ]
            found = [b for b in benchmarks if b.lower() in text.lower()]
            print(f"Potential benchmarks found: {found}")

            # Look for context sentences
            for line in text.split("\n"):
                if (
                    any(b.lower() in line.lower() for b in benchmarks)
                    or "benchmark" in line.lower()
                    or "dataset" in line.lower()
                ):
                    print(line.strip())
    except Exception as e:
        print(f"Error reading {pdf_file}: {e}")
    print("\n")
