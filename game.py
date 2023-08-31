import pygame, sys
from pygame.locals import *
import random
import numpy as np
class game_env:
    def __init__(self,suffix):
        self.alfa = 0.75
        self.beta = 0.75
        self.greedy = 0
        self.random = 1
        self.delta = 0.005
        self.q_table = np.zeros((100,4))
        self.reward_map = {'water.png':-20, 'grass.png':-3,'rock.png':-100,'flowers.png':0,'pig.png':500,'house.png':-500,'already_visited':-10,'invalid':-100}
        self.game_dim = (600,600)
        self.first_cell=0
        self.last_cell=99
        self.text_space=0
        self.initial_cood = (0,0+ self.text_space)
        self.rows,self.columns= 10,10
        self.cell_dim = self.game_dim[0]/self.rows
        self.final_cood = (self.game_dim[0]-self.cell_dim, self.game_dim[1]-self.cell_dim)
        self.game_grid = self.new_game_env()
        #self.game_grid = self.set_load("hgggggggggfgfgfgfgfggfgfgfgfgfwwwgffwwwrrrrrgrrrrrgggrgrffffrffffrgfffggrrrfffrfgggggffggfgggggggggp")
        self.suffix = suffix
        self.action_space = {0:{'x':-1*self.cell_dim,'y':0},
                              2:{'x':self.cell_dim,'y':0}, 
                              1:{'x':0,'y':self.cell_dim},
                              3:{'x':0,'y':-1*self.cell_dim}}
        try:
            with open('env_weights\\weights_{}.npy'.format(self.suffix),'rb') as file:
                                self.q_table = np.load(file)
            with open('env_weights\\env_{}.npy'.format(self.suffix),'rb') as file:
                                self.game_grid = np.load(file)
        except Exception as exp:
            print('No such files pre-exists. Starting a new environment')
            with open('env_weights\\env_{}.npy'.format(self.suffix),'wb') as file:
                                np.save(file,self.game_grid)
            with open('env_weights\\weights_{}.npy'.format(self.suffix),'wb') as file:
                                np.save(file,self.q_table)
            pass 
    def random_board(self):           
         self.game_grid = self.new_game_env()
    def set_load(self,map):
        matrix=[]           
        for i in map:
            cell="grass.png" if i=="g" else "water.png" if i=="w" else "rock.png" if i=="r" else "flowers.png" if i=="f" else "house.png" if i=="h" else "pig.png"
            matrix.append(cell)
        matrix = np.asarray(matrix).reshape(self.rows,self.columns)
        return matrix    


    def new_game_env(self):
        #Crea el tablero
        matrix = random.choices(['grass.png','water.png','rock.png','flowers.png'],weights=[0.35,0.25,0.20,0.20], k=self.rows*self.columns)
        matrix = np.asarray(matrix).reshape(self.rows,self.columns)
        matrix[0][0] ='house.png'
        matrix[0][1]="grass.png"
        matrix[1][1]="flowers.png"
        matrix[1][0]="grass.png"
        matrix[self.rows-1][self.columns-1] = 'pig.png'
        matrix[self.rows-2][self.columns-1] = 'flowers.png'
        matrix[self.rows-2][self.columns-2] = 'flowers.png'
        matrix[self.rows-1][self.columns-2] = 'flowers.png' 
        return matrix    
    def load_images(self,img_path):
        img = pygame.image.load('icons\\{}'.format(img_path))
        img = pygame.transform.scale(img,(self.cell_dim,self.cell_dim))
        return img  
    def print_text(self,text,cood,size):
        font = pygame.font.Font(pygame.font.get_default_font(), size)
        text_surface = font.render(text, True, (255,255,255))
        DISPLAYSURF.blit(text_surface,cood)  
    def initial_state(self):
        DISPLAYSURF.fill((0,0,0))
        for x in range(self.rows):
                for y in range(self.columns):
                    img = self.load_images(self.game_grid[x][y])
                    cood = (y*self.cell_dim,x*self.cell_dim)
                    DISPLAYSURF.blit(img,cood)
        pygame.display.update()
    def steps_visualizer(self,cood):
        img = pygame.image.load('icons\\gnome.png')
        img = pygame.transform.scale(img,(self.cell_dim,self.cell_dim))
        DISPLAYSURF.blit(img,cood)
        pygame.display.update()
        clock.tick(256)      
    def to_cood(self,cood):
        state = int((self.rows*(cood[1]-self.text_space)/self.cell_dim)+(cood[0]/self.cell_dim)) 
        return state
    def to_pygame_cood(self, state):
        cood = int((state%self.rows)*self.cell_dim),int((state//self.rows)*self.cell_dim+self.text_space)
        return cood  
    def is_valid_move(self, cood, already_visited):
        if cood in already_visited:
            return False
        if self.initial_cood[0]<=cood[0]<=self.final_cood[0] and self.initial_cood[1]<=cood[1]<=self.final_cood[1]:
            if self.game_grid[self.to_cood(cood)//self.columns][self.to_cood(cood)%self.rows]=="rock.png":
                return False
            else:
                return True         
        return False        
    def q_table_update(self,  state, action, already_visited):
        curr_cood = self.to_pygame_cood(state)
        new_cood = (int(curr_cood[0] + self.action_space[action]['x']), int(curr_cood[1] + self.action_space[action]['y']))
        new_state = self.to_cood(new_cood)
        is_valid = self.is_valid_move(new_cood, already_visited)                          
        if is_valid:
            reward = self.reward_map[self.game_grid[int(new_state//self.rows)][int(new_state%self.rows)]]
        elif new_cood in already_visited:
            reward = self.reward_map['already_visited']
        else:
            reward = self.reward_map['invalid']               
        try:
            state_value_diff = max(self.q_table[new_state]) - self.q_table[state][action]
        except:
            state_value_diff = 0
        self.q_table[state][action]+=self.alfa*(reward + self.beta*state_value_diff)                                    
        return is_valid, new_state, new_cood
    def episode(self, current_state, is_valid,p):
        #episodio de training
        pygame.event.get()
        cood = self.to_pygame_cood(current_state)
        already_visited = [cood]
        self.steps_visualizer(cood)     
        while current_state!=self.last_cell and is_valid==True:
            #pygame.draw.rect(DISPLAYSURF,(0,0,0),(0,100,self.game_dim[0],50))
            pygame.display.update()
            for event in pygame.event.get():
                if event.type==QUIT:
                                pygame.quit()
                                raise Exception('training ended')
            choice = random.choices([True,False],weights=[self.greedy,self.random],k=1)
            if choice[0]:
                     action = np.argmax(self.q_table[current_state])
            else:
                     action = random.choices([0,1,2,3],weights=[0.25,0.25,0.25,0.25],k=1)
                     action = action[0]
            is_valid, current_state, cood = self.q_table_update(current_state, action, already_visited)        
            pygame.display.update()
            clock.tick(p)
            already_visited.append(cood)
            if is_valid:
                self.steps_visualizer(cood)
            else:
                return (self.to_cood(cood))       
    def episode_no_sprites(self, current_state, is_valid):
        #episodio de training
        cood = self.to_pygame_cood(current_state)
        already_visited = [cood]     
        while current_state!=self.last_cell and is_valid==True:
            choice = random.choices([True,False],weights=[self.greedy,self.random],k=1)
            if choice[0]:
                     action = np.argmax(self.q_table[current_state])
            else:
                     action = random.choices([0,1,2,3],weights=[0.25,0.25,0.25,0.25],k=1)
                     action = action[0]
            is_valid, current_state, cood = self.q_table_update(current_state, action, already_visited)           
            already_visited.append(cood)
            if not is_valid:
                return str(current_state)                
    def training(self, epoch):
                    self.p=48
                    state=random.randint(self.first_cell,self.last_cell)
                    self.initial_state()
                    self.print_text(' Episode:{}'.format(epoch),(180,50),30)
                    lleo=self.episode(state, True,self.p)  
                    print('episode {} ---->'.format(epoch)) 
                    pygame.display.set_caption('greedy={}, random={}'.format(round(self.greedy,4),round(self.random,4)))
                    if epoch%50==0:
                        if self.random>0:
                                self.greedy+=self.delta
                                self.random-=self.delta
                                self.greedy = min(self.greedy,1)
                                self.random= max(self.random,0)                  
                    if epoch%2000==0:
                        self.delta*=2
                        with open('env_weights\\weights_{}.npy'.format(self.suffix),'wb') as f:
                            np.save(f,self.q_table)                 
                    clock.tick(48)
    def training_no_sprites(self, epoch):
                        self.p=8
                        state=random.randint(self.first_cell,self.last_cell)
                        self.episode_no_sprites(state, True)  
                        print('episode {} ---->'.format(epoch))
                        if epoch%50==0:
                            if self.random>0:
                                    self.greedy+=self.delta
                                    self.random-=self.delta
                                    self.greedy = min(self.greedy,1)
                                    self.random= max(self.random,0)                      
                        if epoch%2000==0:
                            self.delta*=2
                            with open('env_weights\\weights_{}.npy'.format(self.suffix),'wb') as f:
                                np.save(f,self.q_table)               
    def testing(self,initial_state=0):
            self.p=8
            self.greedy = 1
            self.random = 0
            #self.print_text(' Episode:{}'.format("test"),(180,50),30)
            with open('env_weights\\env_{}.npy'.format(self.suffix),'rb') as f:
                self.game_grid = np.load(f) 
            with open('env_weights\\weights_{}.npy'.format(self.suffix),'rb') as f:
                self.q_table = np.load(f) 
            self.initial_state()
            lleo=self.episode(initial_state,True,self.p)
            clock.tick(8)
            return lleo      
print("----------------------------------\n Define an existing or new training name: ")
game_env_name=input("")
print("Start training?: (y/n)")
desicion=input()
if desicion.upper()!="N":
    game = game_env(game_env_name)
    pygame.init()
    count=1
    flag = 0
    DISPLAYSURF = pygame.display.set_mode(game.game_dim,0,32)
    clock = pygame.time.Clock()
    while True:
        count+=1
        if count<4900:
            game.training_no_sprites(count)
        else:
            game.training(count)
        if count%5002==0:
            pygame.quit()
            break
        for event in pygame.event.get():
                    if event.type==QUIT:
                                pygame.quit()
                                flag = 1
                                raise Exception('game ended')
        if flag==1:
            break
game = game_env(game_env_name)
flag = 0
pygame.init()
pygame.display.set_caption('Reinforcement Learning')
DISPLAYSURF = pygame.display.set_mode(game.game_dim,0,32)

clock = pygame.time.Clock()
while True:
    game.testing()
    for event in pygame.event.get():
                if event.type==QUIT:
                            pygame.quit()
                            flag = 1
                            raise Exception('game ended')
    if flag==1:
        break