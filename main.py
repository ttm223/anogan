# anoGAN test
import anogan

def main():
    yaml_path = input('input yaml file path')
    ano = anogan.anoGAN
    g_model, d_model = ano.train(yaml_path)

if __name__ == '__main__':
    main()