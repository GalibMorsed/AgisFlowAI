"""Top-level runner for the project.

This file delegates to the `src` package CLI implementation.
"""

def main():
	try:
		from src.main import main as src_main
	except Exception:
		raise

	return src_main()


if __name__ == "__main__":
	raise SystemExit(main())

