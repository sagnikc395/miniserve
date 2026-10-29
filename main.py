## main cli entrypoint for the inference and running everything

import click

from miniserve.demos import attention_demo, kv_demo


@click.group()
def main():
    pass


@main.command()
def serve():
    raise click.ClickException("serve is not implemented yet")


@main.command()
def generate():
    raise click.ClickException("generate is not implemented yet")


@main.command()
def bench():
    raise click.ClickException("bench is not implemented yet")


@main.command('attn-demo')
def attn_demo():
    attention_demo.demo()


@main.command('kv-demo')
def kv_demo_cmd():
    kv_demo.demo()


if __name__ == '__main__':
    main()
