import json
import urllib.request
import os

def main():
    # Read files
    try:
        with open("cs224r_milestone_2026.tex", "r", encoding="utf-8") as f:
            tex_content = f.read()
        with open("cs224r_2026.sty", "r", encoding="utf-8") as f:
            sty_content = f.read()
        with open("reference.bib", "r", encoding="utf-8") as f:
            bib_content = f.read()
    except Exception as e:
        print(f"Error reading files: {e}")
        return

    # Structure payload
    payload = {
        "compiler": "pdflatex",
        "resources": [
            {
                "main": True,
                "content": tex_content
            },
            {
                "path": "cs224r_2026.sty",
                "content": sty_content
            },
            {
                "path": "reference.bib",
                "content": bib_content
            }
        ]
    }

    # Make request
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        "https://latex.ytotech.com/builds/sync",
        data=data,
        headers={"Content-Type": "application/json"}
    )

    print("Sending request to latex.ytotech.com...")
    try:
        with urllib.request.urlopen(req) as response:
            status = response.status
            content_type = response.headers.get("Content-Type", "")
            print(f"Response status: {status}, Content-Type: {content_type}")
            
            if "application/pdf" in content_type:
                pdf_data = response.read()
                with open("cs224r_milestone_2026.pdf", "wb") as f:
                    f.write(pdf_data)
                print("Success! PDF written to cs224r_milestone_2026.pdf")
            else:
                resp_text = response.read().decode("utf-8", errors="ignore")
                print("Error: Did not receive PDF.")
                print(resp_text)
    except urllib.error.HTTPError as e:
        print(f"HTTP Error: {e.code}")
        try:
            error_body = e.read().decode("utf-8", errors="ignore")
            print("Error details:")
            print(error_body)
        except Exception:
            pass
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
