import sys

from markdown_nodes import build_nodes


def main(pdf_path: str) -> None:
    for node in build_nodes(pdf_path):
        print()
        print(node.metadata)
        print(node.get_content())
        print()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: python main.py <pdf_path>")
    main(sys.argv[1])
