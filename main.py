## main cli entrypoint for the inference and running everything

import argparse

from miniserve.demos import attention_demo, kv_demo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('command',choices=['serve','generate','bench','attn-demo','kv-demo'])

    a = ap.parse_args()

    if a.command == "kv-demo":
        kv_demo.demo()
        return
    if a.command == "attn-demo":
        attention_demo.demo()
        return


if __name__ == '__main__':
    main()
