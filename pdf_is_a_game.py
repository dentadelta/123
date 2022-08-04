#turn each image patch into a sprite object, making them fly randomly on the pdf/screen
import pygame
import json
import random
import time

Jsonfile = '/home/delta/vscode/dataextraction/croppedData/croppedData04_08_2022_02_03_22.json'

RED = (255, 0, 0)
factor = 3
def random_():
    return random.randrange(-1,1)/10

class Sprite(pygame.sprite.Sprite):
    def __init__(self, color,imagepath):
        super().__init__()
        image = pygame.image.load(imagepath)
        width = image.get_width()
        height = image.get_height()
        self.image = pygame.transform.scale(image, (width/factor, height/factor))
        pygame.draw.rect(self.image,color,pygame.Rect(0, 0, 0, 0))
        self.rect = self.image.get_rect()
        self.mask = pygame.mask.from_surface(self.image)

    def update(self):
        time.sleep(0.01)
        self.rect.center = 0.1*self.rect.x +random_(), 0.1*self.rect.y+random_()
        if self.rect.x < 0:
            self.rect.x = random.randrange(0, int(332*5/factor))
        if self.rect.y < 0:
            self.rect.y += random.randrange(0, int(468*5/factor))

def loadSprite(imagepath,x,y):
    object_ = Sprite(RED, imagepath)
    object_.rect.x = x/factor
    object_.rect.y = y/factor
    return object_
with open(Jsonfile,'r') as jsonfile:
    jsondata = json.load(jsonfile)
    jsonfile.close()

Imagepath = []
X = []
Y = []
for k,v in jsondata.items():
    save_folder = v['savefolder']
    box = v['box']
    Imagepath.append(f'/home/delta/vscode/dataextraction/{save_folder}')
    X.append(box[0])
    Y.append(box[1])

        
pygame.init()
pygame.display.set_caption("pdf to game")
clock = pygame.time.Clock()
screen = pygame.display.set_mode((332*5/factor, 468*5/factor), pygame.RESIZABLE)
running = True
all_sprites_list = pygame.sprite.Group()
for i in range(len(Imagepath)):
    object_ = loadSprite(Imagepath[i],X[i],Y[i])
    all_sprites_list.add(object_)
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    clock.tick(30)
    all_sprites_list.update()
    screen.blit(pygame.image.load("blank.png"), (0, 0))
    screen.fill((255, 255, 255))
    all_sprites_list.draw(screen)
    pygame.display.flip()
pygame.quit()
