from cfie_training.cli.main import main


if __name__ == "__main__":
    # 通过包级入口转发到训练 CLI 主函数。
    raise SystemExit(main())
