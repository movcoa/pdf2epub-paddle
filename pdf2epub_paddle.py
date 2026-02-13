import os
import sys
import argparse
import base64
import json
import re
import tempfile


import requests
import fitz  # PyMuPDF
from ebooklib import epub
from typing import List, Dict, Any
from dotenv import load_dotenv

# Load .env file if present (does not override existing env vars)
load_dotenv()

# --- Configuration ---
API_URL = "https://d8y5fdsclb20d640.aistudio-app.com/layout-parsing"
# Environment variable for API token
API_TOKEN = os.getenv("PADDLE_API_TOKEN", "")

CHUNK_SIZE = 5  # Reduced to 5 for maximum reliability
MAX_DAILY_PAGES = 3000


def check_dependencies():
    """Checks if required libraries are installed."""
    missing = []
    try:
        import fitz
    except ImportError:
        missing.append("pymupdf")
    try:
        import ebooklib
    except ImportError:
        missing.append("EbookLib")

    if missing:
        print(f"[!] Missing dependencies: {', '.join(missing)}")
        print(f"    Please run: pip install {' '.join(missing)}")
        return False
    return True


def split_pdf(file_path: str, chunk_size: int = CHUNK_SIZE) -> List[str]:
    """
    Splits a PDF into chunks of `chunk_size` pages.
    Returns a list of paths to the temporary chunk files.
    """
    doc = fitz.open(file_path)
    total_pages = len(doc)
    print(f"[*] Total pages: {total_pages}")

    if total_pages > MAX_DAILY_PAGES:
        print(
            f"[!] WARNING: This document ({total_pages} pages) exceeds the daily API limit of {MAX_DAILY_PAGES} pages."
        )
        print("    Processing may fail or get blocked if you exceed your quota.")

    chunk_paths = []
    temp_dir = tempfile.mkdtemp(prefix="pdf_chunks_")

    for start_page in range(0, total_pages, chunk_size):
        end_page = min(start_page + chunk_size, total_pages)
        # Create a new PDF for this chunk
        chunk_doc = fitz.open()
        chunk_doc.insert_pdf(doc, from_page=start_page, to_page=end_page - 1)

        chunk_filename = os.path.join(temp_dir, f"chunk_{start_page}_{end_page}.pdf")
        chunk_doc.save(chunk_filename)
        chunk_doc.close()
        chunk_paths.append(chunk_filename)

    doc.close()
    return chunk_paths


def parse_pdf_chunk(chunk_path: str, token: str) -> Dict[str, Any]:
    """
    Sends a PDF chunk to the PaddleOCR API and returns the parsed result.
    """
    print(f"[*] uploading chunk: {os.path.basename(chunk_path)}")

    with open(chunk_path, "rb") as file:
        file_bytes = file.read()
        file_data = base64.b64encode(file_bytes).decode("ascii")

    headers = {"Authorization": f"token {token}", "Content-Type": "application/json"}

    payload = {
        "file": file_data,
        "fileType": 0,  # 0 for PDF
        "useDocOrientationClassify": False,
        "useDocUnwarping": False,
        "useChartRecognition": False,  # Basic extraction
    }

    max_retries = 5
    for attempt in range(max_retries):
        try:
            response = requests.post(
                API_URL, json=payload, headers=headers, timeout=180
            )
            response.raise_for_status()
            result = response.json()

            # Handle API responses safely
            # The API seems to return 'result' directly in some cases, or an 'error' field
            if "error" in result:
                print(
                    f"[!] API Error for chunk {os.path.basename(chunk_path)}: {result['error']}"
                )
                return None

            # If 'responses' is missing but 'result' is present, adapt the result
            if "responses" not in result and "result" in result:
                # Wrap the result to maintain compatibility with the rest of the script
                # if possible, or just return the result directly if it's already the expected object.
                pass

            return process_layout_results(result, chunk_path)

        except (requests.exceptions.RequestException, ConnectionError) as e:
            wait_time = (
                2**attempt
            ) * 5  # Exponential backoff: 5, 10, 20, 40, 80 seconds
            print(f"[!] API Request failed (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                print(f"    Retrying in {wait_time}s...")
                import time

                time.sleep(wait_time)
            else:
                print(
                    f"[!] Permanently failed processing chunk: {os.path.basename(chunk_path)}"
                )
                return None
        except Exception as e:
            print(
                f"[!] Unexpected error processing chunk {os.path.basename(chunk_path)}: {e}"
            )
            return None


def process_layout_results(result, chunk_path):
    """Helper to return the result if the original function is missing."""
    return result


def extract_cover_image(pdf_path: str, output_path: str) -> str:
    """Renders the first page of a PDF as a PNG image for use as an EPUB cover."""
    doc = fitz.open(pdf_path)
    try:
        if len(doc) == 0:
            print("[!] PDF has no pages; skipping cover extraction.")
            return None
        page = doc.load_page(0)
        mat = fitz.Matrix(2, 2)  # 2x zoom (~144 DPI)
        pix = page.get_pixmap(matrix=mat)
        pix.save(output_path)
        print(f"[*] Cover image extracted to {output_path}")
        return output_path
    except Exception as e:
        print(f"[!] Failed to extract cover image: {e}")
        return None
    finally:
        doc.close()


def extract_metadata_interactive(results: List[Dict], default_title: str) -> Dict[str, str]:
    """Shows the first page OCR text and prompts the user for title and author."""
    first_page_text = ""
    try:
        first_page_text = results[0]["result"]["layoutParsingResults"][0]["markdown"]["text"]
    except (IndexError, KeyError, TypeError):
        pass

    if first_page_text:
        print("\n--- First page OCR text ---")
        print(first_page_text.strip())
        print("----------------------------\n")
    else:
        print("\n[!] Could not extract text from the first page.\n")

    title = input(f"Enter book title (or press Enter to use '{default_title}'): ").strip()
    if not title:
        title = default_title

    author = input("Enter author name (or press Enter to skip): ").strip()

    return {"title": title, "author": author if author else None}


def extract_candidate_headings(results: List[Dict]) -> List[Dict]:
    """Scans API results and extracts candidate chapter headings with page numbers."""
    candidates = []
    any_header_pattern = re.compile(r"^(#{1,2})\s+(.+)$")
    latex_pattern = re.compile(r"\$\s*\\underline\{(.+?)\}\s*\$")
    global_page = 0

    for result in results:
        if not result or "result" not in result:
            continue
        for page_res in result["result"].get("layoutParsingResults", []):
            global_page += 1
            page_md = page_res["markdown"]["text"]
            for line in page_md.split("\n"):
                if line.strip().isdigit():
                    continue
                match = any_header_pattern.match(line)
                if match:
                    title = match.group(2).strip()
                    # Clean LaTeX artifacts (e.g. "$ \underline{3} $ Title" -> "Title")
                    title = latex_pattern.sub("", title).strip()
                    # Strip leading bare numbers left after LaTeX cleanup (e.g. "3 What Is..." -> "What Is...")
                    title = re.sub(r"^\d+\s+", "", title).strip()
                    candidates.append({
                        "title": title,
                        "page": global_page,
                        "level": len(match.group(1)),
                        "md_line": line,
                    })
    return candidates


def filter_heading_candidates(candidates: List[Dict]) -> List[Dict]:
    """Adaptively filters headings by trying H1-only, then keyword match, then all."""
    chapter_keyword = re.compile(
        r"^(?:Chapter|Part|Lecture|Preface|Intro|Appendix|Prologue|Epilogue|"
        r"Conclusion|Acknowledgements|Contents|Abstract|序|前言|导论|目录|附录|后记|"
        r"第[零一二三四五六七八九十百千0-9]+[篇章节讲])",
        re.IGNORECASE,
    )

    # Strategy 1: H1-only (skip front-matter on first 2 pages)
    h1 = [h for h in candidates if h["level"] == 1 and h["page"] > 2]
    if len(h1) >= 4:
        return h1

    # Strategy 2: Keyword-matched headings (any level)
    keyword_matches = [h for h in candidates if chapter_keyword.match(h["title"])]
    if len(keyword_matches) >= 3:
        return keyword_matches

    # Strategy 3: All headings (last resort)
    return candidates


def review_toc_interactive(candidates: List[Dict], all_candidates: List[Dict] = None) -> List[Dict]:
    """Interactive prompt for reviewing and editing the auto-detected TOC."""
    if not candidates:
        print("\n[!] No chapter headings detected. The book will be a single chapter.")
        return []

    def _print_heading_list(headings):
        print(f"  {'#':>3}  | {'Pg':>4} | Heading")
        print(f"  {'---':>3}--+------+{'-' * 40}")
        for i, h in enumerate(headings):
            indent = "  " if h["level"] == 1 else "    "
            print(f"  {i+1:>3}  | {h['page']:>4} | {indent}{h['title']}")

    def _show_options():
        print()
        print("Options:")
        print("  [Enter]    Accept all headings as chapter split points")
        print("  1,3,5      Remove headings by number (comma-separated)")
        print("  +1,3,5     Keep ONLY these headings (comma-separated)")
        if all_candidates and len(all_candidates) > len(candidates):
            print(f"  all        Show all {len(all_candidates)} headings (including sub-sections)")
        print("  none       No chapters (entire book as single chapter)")
        print()

    print("\n--- Detected Chapter Headings ---")
    _print_heading_list(candidates)
    _show_options()

    while True:
        choice = input("Your choice: ").strip()

        if choice == "":
            return list(candidates)
        elif choice.lower() == "none":
            return []
        elif choice.lower() == "all" and all_candidates and len(all_candidates) > len(candidates):
            candidates = all_candidates
            all_candidates = None
            print(f"\n--- All Headings ({len(candidates)}) ---")
            _print_heading_list(candidates)
            _show_options()
            continue
        else:
            try:
                # Detect keep mode (+) vs remove mode (default)
                keep_mode = choice.startswith("+")
                if keep_mode:
                    choice = choice[1:]  # strip the +

                nums = set()
                valid = True
                for part in choice.split(","):
                    part = part.strip().lstrip("-")  # allow optional - prefix
                    num = int(part)
                    if 1 <= num <= len(candidates):
                        nums.add(num)
                    else:
                        print(f"  [!] Invalid number: {num} (must be 1-{len(candidates)})")
                        valid = False
                        break

                if not valid:
                    continue

                if keep_mode:
                    confirmed = [h for i, h in enumerate(candidates) if (i + 1) in nums]
                else:
                    confirmed = [h for i, h in enumerate(candidates) if (i + 1) not in nums]

                if confirmed:
                    print(f"\nUpdated TOC ({len(confirmed)} chapters):")
                    _print_heading_list(confirmed)
                else:
                    print("\n  All headings removed. Book will be a single chapter.")

                confirm = input("\nConfirm? [Y/n]: ").strip().lower()
                if confirm in ("", "y", "yes"):
                    return confirmed
                else:
                    print("\n--- Detected Chapter Headings ---")
                    _print_heading_list(candidates)
                    _show_options()
                    continue

            except ValueError:
                print("  [!] Invalid input. Use numbers like '1,3,5' or '+1,3,5' for keep mode.")
                continue


def download_image(url: str, save_path: str):
    """Downloads an image from a URL to a local path."""
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, "wb") as f:
                f.write(response.content)
            return True
    except Exception as e:
        print(f"[!] Failed to download image {url}: {e}")
    return False


def create_epub(title: str, results: List[Dict], output_file: str, image_dir: str,
                cover_image_path: str = None, author: str = None,
                confirmed_headings: List[Dict] = None):
    """
    Creates an EPUB file from the aggregated API results.
    """
    book = epub.EpubBook()
    book.set_identifier(f"id_{title}")
    book.set_title(title)
    book.set_language("en")  # Or auto-detect?

    if author:
        book.add_author(author)

    # Set cover image
    if cover_image_path and os.path.exists(cover_image_path):
        with open(cover_image_path, "rb") as f:
            cover_data = f.read()
        book.set_cover("cover.png", cover_data)

    chapters = []

    # CSS for the book
    style = """
    body { font-family: serif; line-height: 1.8; text-align: justify; margin: 1em; }
    h1 { text-align: center; margin: 1.5em 0 0.8em 0; font-size: 1.6em; }
    h2 { margin: 1.2em 0 0.6em 0; font-size: 1.3em; }
    h3 { margin: 1em 0 0.5em 0; font-size: 1.1em; }
    p { margin-bottom: 0.8em; }
    blockquote { margin: 1em 2em; font-style: italic; }
    img { max-width: 100%; height: auto; display: block; margin: 1em auto; }
    """
    nav_css = epub.EpubItem(
        uid="style_nav", file_name="style/nav.css", media_type="text/css", content=style
    )
    book.add_item(nav_css)

    # Process each chunk's result
    # The API returns 'result' -> 'layoutParsingResults' (list)
    # Each item in layoutParsingResults corresponds to a page (usually)

    # Combine all pages into chapters if they have headings, or just page-by-page?
    # Better to just append text. If we see a H1/H2, we can make a split potentially,
    # but for simplicity, let's dump pages into one big flow or chapter-per-chunk?
    # 50 pages per chapter is too much.
    # Let's try to detect headers in the markdown to split chapters.

    full_markdown = ""

    # Aggregated images to add to the book
    # Map API image path (e.g. 'images/tmp.jpg') to internal EPUB path

    for chunk_idx, result in enumerate(results):
        if not result or "result" not in result:
            continue

        layout_results = result["result"].get("layoutParsingResults", [])

        for i, page_res in enumerate(layout_results):
            # 1. Get Markdown Text
            page_md = page_res["markdown"]["text"]

            # 2. Handle Images
            # page_res["markdown"]["images"] is { relative_path: url }
            # The markdown text uses relative_path: ![](images/xxx.jpg)
            images_map = page_res["markdown"].get("images", {})

            for rel_path, img_url in images_map.items():
                # Define local path where we downloaded the image
                local_img_path = os.path.join(image_dir, rel_path)

                # Check if we successfully downloaded it
                if os.path.exists(local_img_path):
                    # Add to EPUB
                    # Read image data
                    with open(local_img_path, "rb") as img_f:
                        img_data = img_f.read()

                    # Create EPUB Image Item
                    # Use the same relative path as filename to match markdown links
                    # (assuming markdown is ![](images/xxx))
                    epub_img = epub.EpubImage()
                    epub_img.file_name = rel_path
                    epub_img.media_type = (
                        "image/jpeg"  # Assuming JPG, might need detection
                    )
                    epub_img.content = img_data

                    if rel_path not in [item.file_name for item in book.get_items()]:
                        book.add_item(epub_img)

            # 3. Concatenate Text with paragraph reflow
            # Reflow lines into continuous paragraphs to fix sentence splitting issues
            # Pages often break sentences mid-flow, and OCR can split paragraphs
            # into separate lines that should be joined.
            
            lines = page_md.split("\n")
            cleaned_lines = []
            for line in lines:
                # Basic Page Number Removal (digit only)
                if line.strip().isdigit():
                    continue
                cleaned_lines.append(line)
            
            # Reflow: join lines that appear to be part of the same paragraph
            reflowed_paragraphs = []
            current_para = ""
            
            # Terminal punctuation that indicates end of sentence/paragraph
            terminal_chars = ('.', '!', '?', '"', "'", ')', ']', '}', ':', ';')
            # Patterns that indicate a line is a header or special block
            header_pattern = re.compile(r'^(#{1,6}\s|>|\s*[-*]\s|\d+\.)')
            # Characters that indicate a line should not be joined with next
            non_join_endings = (':', ';', ',', '-')
            
            for i, line in enumerate(cleaned_lines):
                stripped = line.rstrip()
                if not stripped:
                    # Empty line indicates paragraph break
                    if current_para:
                        reflowed_paragraphs.append(current_para)
                        current_para = ""
                    continue
                
                # Check if this line is a header/list/blockquote
                is_special = header_pattern.match(stripped)
                
                if is_special:
                    # Flush current paragraph before special line
                    if current_para:
                        reflowed_paragraphs.append(current_para)
                        current_para = ""
                    reflowed_paragraphs.append(stripped)
                elif not current_para:
                    # Start of new paragraph
                    current_para = stripped
                else:
                    # Check if current_para ends with terminal punctuation
                    # or if it ends with colon/comma/semicolon (likely list/address item)
                    current_ends = current_para.rstrip()
                    
                    if current_ends.endswith(terminal_chars):
                        # Previous paragraph ends with terminal punctuation
                        # This line starts a new paragraph
                        reflowed_paragraphs.append(current_para)
                        current_para = stripped
                    elif current_ends and current_ends[-1] in non_join_endings:
                        # Previous line ends with :, ;, , or - - likely list item or address
                        # Don't join, start new paragraph
                        reflowed_paragraphs.append(current_para)
                        current_para = stripped
                    elif len(stripped) < 20 and not stripped.endswith(terminal_chars):
                        # Short line that doesn't end with punctuation - might be 
                        # a title, caption, or deliberate short line (poetry, etc.)
                        reflowed_paragraphs.append(current_para)
                        current_para = stripped
                    else:
                        # Likely continuation of same paragraph
                        current_para = current_ends + " " + stripped
            
            # Don't forget the last paragraph
            if current_para:
                reflowed_paragraphs.append(current_para)
            
            # Join paragraphs and append to full_markdown
            # But first, check if we need to merge with the previous page's content
            # (sentences can be split across page boundaries)
            page_markdown = "\n\n".join(reflowed_paragraphs)
            
            if full_markdown and page_markdown:
                # Check if full_markdown ends mid-sentence (no terminal punctuation)
                # and page_markdown starts with a continuation (not a header)
                full_stripped = full_markdown.rstrip()
                page_lines = page_markdown.split('\n')
                first_page_line = page_lines[0].strip() if page_lines else ""
                
                # Check if we should merge (previous doesn't end with terminal, next isn't a header)
                ends_mid_sentence = full_stripped and not full_stripped.endswith(terminal_chars)
                next_is_header = header_pattern.match(first_page_line) if first_page_line else False
                
                if ends_mid_sentence and not next_is_header:
                    # Merge: remove the trailing newlines and join with space
                    full_markdown = full_markdown.rstrip() + " " + page_markdown + "\n\n"
                else:
                    full_markdown += page_markdown + "\n\n"
            else:
                full_markdown += page_markdown + "\n\n"

    # Split markdown into chapters based on headers (# Header)
    # If no headers found, put everything in one chapter.

    # Split by H1 or H2, BUT only if it looks like a real Chapter/Part title
    # Regex for headers: ^#+ \s* (Title)

    md_lines = full_markdown.split("\n")
    current_chapter_title = "Start"
    current_chapter_content = []
    chapter_count = 0

    # Regex to identify "Major" headers (Chapters/Parts) to split on (legacy fallback)
    major_header_pattern = re.compile(
        r"^(#{1,2})\s+(?:Chapter|Part|Lecture|Preface|Intro|Appendix|Prologue|Epilogue|Conclusion|Book|Acknowledgements|Contents|Abstract|序|前言|导论|目录|第[零一二三四五六七八九十百千0-9]+[篇章讲]).*"
    )

    # Regex for ANY header to format as H1/H2 in HTML but not necessarily split
    any_header_pattern = re.compile(r"^(#{1,2})\s+(.+)$")

    # Build set of confirmed heading titles for O(1) lookup
    if confirmed_headings is not None:
        confirmed_titles = {h["title"] for h in confirmed_headings}
    else:
        confirmed_titles = None  # use legacy regex fallback

    for line in md_lines:
        # Check if line is a header (before LaTeX cleanup, to match extracted candidates)
        match = any_header_pattern.match(line)
        is_split_point = False
        if match:
            heading_text = match.group(2).strip()
            # Clean LaTeX artifacts for matching
            heading_text = re.sub(r"\$\s*\\underline\{(.+?)\}\s*\$", "", heading_text).strip()
            heading_text = re.sub(r"^\d+\s+", "", heading_text).strip()
            # Determine if this heading is a chapter split point
            if confirmed_titles is not None:
                is_split_point = heading_text in confirmed_titles
            else:
                is_split_point = bool(major_header_pattern.match(line))

        # Clean up LaTeX artifacts
        line = re.sub(r"\$\s*\^\{(.+?)\}\s*\$", r"", line)  # superscripts
        line = re.sub(r"\$\s*\\underline\{(.+?)\}\s*\$", "", line)  # underlines

        if match:
            if is_split_point:
                # If we have content for the previous chapter, save it
                if current_chapter_content:
                    # Determine filename
                    safe_title = "".join(
                        [
                            c
                            for c in current_chapter_title
                            if c.isalnum() or c in (" ", "_", "-")
                        ]
                    ).strip()
                    if not safe_title:
                        safe_title = f"chap_{chapter_count}"

                    c = epub.EpubHtml(
                        title=current_chapter_title,
                        file_name=f"{safe_title}_{chapter_count}.xhtml",
                        lang="en",
                    )

                    try:
                        import markdown

                        html_content = markdown.markdown(
                            "\n".join(current_chapter_content)
                        )
                    except ImportError:
                        html_content = (
                            "<p>" + "</p><p>".join(current_chapter_content) + "</p>"
                        )

                    c.content = f"<html><head><link rel='stylesheet' href='style/nav.css'/></head><body>{html_content}</body></html>"
                    c.add_item(nav_css)
                    book.add_item(c)
                    chapters.append(c)
                    chapter_count += 1

                current_chapter_title = match.group(2)
                current_chapter_content = [line]
            else:
                # It's a minor header (e.g. Section), just keep it in flow
                current_chapter_content.append(line)
        else:
            current_chapter_content.append(line)

    # Add last chapter
    if current_chapter_content:
        # Determine filename
        safe_title = "".join(
            [c for c in current_chapter_title if c.isalnum() or c in (" ", "_", "-")]
        ).strip()
        if not safe_title:
            safe_title = f"chap_{chapter_count}"

        c = epub.EpubHtml(
            title=current_chapter_title,
            file_name=f"{safe_title}_{chapter_count}.xhtml",
            lang="en",
        )
        try:
            import markdown

            html_content = markdown.markdown("\n".join(current_chapter_content))
        except ImportError:
            html_content = "<p>" + "</p><p>".join(current_chapter_content) + "</p>"

        c.content = f"<html><head><link rel='stylesheet' href='style/nav.css'/></head><body>{html_content}</body></html>"
        c.add_item(nav_css)
        book.add_item(c)
        chapters.append(c)

    # If no chapters were created (no headers found), create one big chapter
    if not chapters and full_markdown:
        c = epub.EpubHtml(title="Content", file_name="content.xhtml", lang="en")
        try:
            import markdown

            html_content = markdown.markdown(full_markdown)
        except ImportError:
            html_content = "<p>" + "</p><p>".join(full_markdown.split("\n")) + "</p>"
        c.content = f"<html><head><link rel='stylesheet' href='style/nav.css'/></head><body>{html_content}</body></html>"
        book.add_item(c)
        chapters.append(c)

    book.toc = tuple(chapters)
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())
    book.spine = ["nav"] + chapters

    epub.write_epub(output_file, book, {})
    print(f"[*] EPUB saved to {output_file}")


def main():
    if not check_dependencies():
        return

    parser = argparse.ArgumentParser(description="Scanned PDF to Epub Converter")
    parser.add_argument("input_pdf", help="Path to input PDF file")
    parser.add_argument(
        "--output", "-o", help="Path to output EPUB file (default: input_name.epub)"
    )
    parser.add_argument("--title", help="Book title (skips interactive prompt)")
    parser.add_argument("--author", help="Author name (skips interactive prompt)")
    parser.add_argument("--auto-toc", action="store_true",
                        help="Skip interactive TOC review; use auto-detected headings")
    parser.add_argument("--no-toc", action="store_true",
                        help="Skip heading detection; produce single-chapter EPUB")
    args = parser.parse_args()

    input_path = args.input_pdf
    if not os.path.exists(input_path):
        print(f"Error: File {input_path} not found.")
        return

    if args.auto_toc and args.no_toc:
        print("[!] Error: --auto-toc and --no-toc are mutually exclusive.")
        return

    if not API_TOKEN:
        print("[!] Error: PADDLE_API_TOKEN environment variable is not set.")
        print("    Please set it using: export PADDLE_API_TOKEN='your_token_here'")
        return

    if not args.output:
        args.output = os.path.splitext(input_path)[0] + ".epub"

    # Create a unique work directory based on the input filename hash
    import hashlib

    file_hash = hashlib.md5(os.path.basename(input_path).encode("utf-8")).hexdigest()[
        :8
    ]
    work_dir = f"paddle_epub_work_{file_hash}"

    print(f"[*] Work directory: {work_dir}")

    image_dir = os.path.join(work_dir, "images")
    os.makedirs(image_dir, exist_ok=True)

    try:
        # Step 1: Chunking
        print("[-] Step 1: Splitting PDF...")
        chunk_paths = split_pdf(input_path)

        # Step 1.5: Extract cover image
        cover_path = extract_cover_image(input_path, os.path.join(work_dir, "cover.png"))

        # Step 2: API Processing
        results = []
        print(f"[-] Step 2: Processing {len(chunk_paths)} chunks via PaddleOCR API...")
        for i, chunk in enumerate(chunk_paths):
            chunk_name = os.path.basename(chunk)
            json_checkpoint = os.path.join(work_dir, chunk_name + ".json")

            if os.path.exists(json_checkpoint):
                print(
                    f"    [+] Resuming: Found checkpoint for chunk {i + 1}/{len(chunk_paths)}"
                )
                with open(json_checkpoint, "r") as f:
                    res = json.load(f)
            else:
                # Rate limiting: Sleep before new request
                if i > 0:
                    print("    ...waiting 5s to respect API rate limits...")
                    import time

                    time.sleep(5)

                print(f"    Processing chunk {i + 1}/{len(chunk_paths)}...")
                # Increased timeout to 180s
                res = parse_pdf_chunk(chunk, API_TOKEN)

                if res:
                    # Save checkpoint
                    os.makedirs(os.path.dirname(json_checkpoint), exist_ok=True)
                    with open(json_checkpoint, "w") as f:
                        json.dump(res, f)

            if res:
                results.append(res)

                # Download images immediately to save locally
                layout_results = res.get("result", {}).get("layoutParsingResults", [])
                for page_res in layout_results:
                    images_map = page_res["markdown"].get("images", {})
                    for rel_path, img_url in images_map.items():
                        local_path = os.path.join(image_dir, rel_path)
                        if not os.path.exists(
                            local_path
                        ):  # Don't re-download if exists
                            download_image(img_url, local_path)
            else:
                print(
                    f"[!] CRITICAL: Failed to process chunk {i + 1}. Aborting to prevent incomplete book."
                )
                print(
                    "    Please resolve connectivity issues and re-run the script to resume."
                )
                sys.exit(1)

        # Step 2.5: Metadata extraction
        default_title = os.path.splitext(os.path.basename(input_path))[0]
        if args.title:
            metadata = {"title": args.title, "author": args.author}
        else:
            metadata = extract_metadata_interactive(results, default_title)

        # Step 2.75: TOC Review
        if args.no_toc:
            confirmed_headings = []
        elif args.auto_toc:
            confirmed_headings = None
        else:
            print("[-] Step 2.75: Detecting chapter headings...")
            candidates = extract_candidate_headings(results)
            filtered = filter_heading_candidates(candidates)
            if len(candidates) > len(filtered):
                print(f"  ({len(filtered)} chapter headings found, {len(candidates) - len(filtered)} sub-headings hidden)")
            confirmed_headings = review_toc_interactive(filtered, all_candidates=candidates)

        # Step 3: Generation
        print("[-] Step 3: Generating EPUB...")
        create_epub(metadata["title"], results, args.output, image_dir,
                    cover_image_path=cover_path, author=metadata["author"],
                    confirmed_headings=confirmed_headings)

    finally:
        # Cleanup temp chunks
        # shutil.rmtree(work_dir) # Optional: keep for debugging?
        # Let's cleanup the split PDFs but maybe keep the images for a bit or just assume user might want them?
        # For a clean tool, we should clean up.
        # But for now, let's leave 'paddle_epub_work' so user can inspect if something goes wrong.
        print(
            f"[*] Done. Intermediate files are in '{work_dir}'. You can delete this folder if verified."
        )


if __name__ == "__main__":
    main()
