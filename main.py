# anoGAN test
import anogan


def main():
    # now testing condition
    yaml_path = './params.yaml'
    # yaml_path = input('input yaml file path: ')
    print('get path: {}'.format(yaml_path))
    ano = anogan.anoGAN()
    _, _ = ano.train(yaml_path=yaml_path)

if __name__ == '__main__':
    main()