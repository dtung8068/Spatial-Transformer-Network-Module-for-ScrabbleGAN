from PIL import Image
import torch
import numpy as np
from options.test_options import TestOptions
from models.BigGAN_networks import Generator
from util.util import *

def load_model():
    opt = TestOptions().parse()  # get test options
    seed_rng(opt.seed)
    opt.n_classes = 80
    gen = Generator(**vars(opt))

    state_dict = torch.load('checkpoints/stnCVL_CVLcharH32W16_CapitalizeLex_GANres16_bs8_useRNN/100_net_G.pth')
    if hasattr(state_dict, '_metadata'):
        del state_dict._metadata
    gen.load_state_dict(state_dict)
    gen.cuda()
    gen.eval()
    return gen

model = load_model()


alphabets_english = 'Only thewigsofrcvdampbkuq.A-210xT5\'MDL,RYHJ"ISPWENj&BC93VGFKz();#:!7U64Q8?+*ZX/%'
char_to_int = dict()
for ind, c in enumerate(alphabets_english):
    char_to_int[c] = ind


def get_word(word):
    encoded = [char_to_int[char] for char in word]
    words = torch.zeros((1, len(encoded), 80), dtype=torch.int32)
    for i, code in enumerate(encoded):
        words[0, i, code] = 1
    return words

def generate_image(word):
    words = get_word(word)
    z, _ = prepare_z_y(1, 128, 80, device=torch.device('cuda'))
    res = model.forward(z=z, y=words.to(torch.device('cuda')).type(torch.float32)).cpu()
    res = res.detach().numpy()[0, 0] * 255
    im = Image.fromarray(res).convert('RGB')
    return im

i = 1
#with open('imgs/cvl_labels.txt', 'r') as file:
#    for line in file:
#        for word in line.split():
#            img = generate_image(word)
#            image = Image.new('RGB', (16 * 13, 32), (255, 255, 255))
#            image.paste(img, (0, 0))
#            image.save('./imgs/stn_cvl_test.png'.format(i))
#            i += 1
img = generate_image("swimming")
#image = Image.new('RGB', (16 * 13, 32), (255, 255, 255))
#image.paste(img, (0, 0))
img.save('./imgs/stn_cvl_test_2.png'.format(i))
i += 1