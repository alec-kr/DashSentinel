#!/usr/bin/env python3
from src.app import DashSentinelApp
from src.cli import parse_args


def main():
    """Main function to run the app"""
    args = parse_args()
    app = DashSentinelApp(args)
    app.run()


if __name__ == "__main__":
    main()
