import pygame, sys
from pygame.locals import *
import random
import numpy as np
from collections import Counter

class game_env:
    def __init__(self,size):
        self.rows,self.columns= size,size
        self.initial_cood = [0,0]
        self.rates_table = np.zeros(self.rows**2)
        self.game_dim = (600,600)
        self.first_cell=0
        self.last_cell=16
        self.text_space=0
        self.cell_dim = self.game_dim[0]/self.rows
        self.final_cood = (self.game_dim[0]-self.cell_dim, self.game_dim[1]-self.cell_dim)
        self.tiles=list(range(self.rows**2))
        self.tiles.remove(self.to_cood(self.initial_cood))
        self.game_list = self.new_game_env()
        self.action_space = {0:{'x':-1*self.cell_dim,'y':0}, 
                              2:{'x':self.cell_dim,'y':0}, 
                              1:{'x':0,'y':self.cell_dim},
                              3:{'x':0,'y':-1*self.cell_dim}}  
        self.wumpus=True
        self.coods=self.initial_cood
        self.safe_squares=[]
        self.already_visited=[]

        self.alfa = 0.75
        self.beta = 0.75
        self.greedy = 0
        self.random = 1
        self.delta = 0.005
        self.q_table = np.zeros((self.rows**2,4))
        self.reward_map = {'square.png':-10,'pit.png':-1000,'gold.png':1000,'already_visited':-10,'invalid':-1000}


    def new_game_env(self): #Crea el tablero
        level = random.choices(['square.png','pit.png'],weights=[0.8,0.2], k=self.rows*self.columns)
        level[0] ='square.png'
        self.gold_state=random.choice(self.tiles)
        self.tiles.remove((self.gold_state))
        self.wump_state=random.choice(self.tiles)
        return level
    
    def display_smth(self, cood, file):
        img = pygame.transform.scale(pygame.image.load(file),(self.cell_dim,self.cell_dim))
        DISPLAYSURF.blit(img,cood)

    def print_text(self,text,cood,size):
        font = pygame.font.Font(pygame.font.get_default_font(), size)
        text_surface = font.render(text, True, (255,255,255))
        DISPLAYSURF.blit(text_surface,cood)

    def initial_state(self):
        DISPLAYSURF.fill((0,0,0))
        for x in range(self.rows):
                for y in range(self.columns):
                    self.display_smth(cood=(y*self.cell_dim,x*self.cell_dim), file=self.game_list[x*self.rows + y])
                    self.display_smth(cood=self.to_pygame_cood(self.gold_state), file="gold.png")
                    if self.wumpus:
                          self.display_smth(cood=self.to_pygame_cood(self.wump_state), file="wumpus.png")
                    self.display_smth(cood=self.coods, file="agent .png")

    def steps_visualizer(self,cood):
        img = pygame.image.load('agent .png')
        img = pygame.transform.scale(img,(self.cell_dim,self.cell_dim))
        DISPLAYSURF.blit(img,cood)
        pygame.display.update()
        clock.tick(256)    
    
    def to_cood(self,cood):
        state = int((self.rows*(cood[1])/self.cell_dim)+(cood[0]/self.cell_dim)) 
        return state
    
    def to_pygame_cood(self, state):
        cood = int((state%self.rows)*self.cell_dim),int((state//self.rows)*self.cell_dim)
        return cood
      
    def is_valid_move(self, cood):

        if self.initial_cood[0]<=cood[0]<=self.final_cood[0] and self.initial_cood[1]<=cood[1]<=self.final_cood[1]:
            return True         
        return False
              
    
    def check_moves(self,valid_coods): #v_coods in self.to_pygame_cood
        safePit=True
        for v_c in valid_coods:
            if self.game_list[v_c]=="pit.png" and safePit:
                  safePit=False
                  self.display_smth(self.coods, "breeze.png")
                  for i in valid_coods:
                       if i not in self.safe_squares:
                            self.rates_table[i]-=10
                  break
        safeWumpus=True
        for v_c in valid_coods:
            if v_c==self.wump_state and safeWumpus:
                safeWumpus=False

                self.display_smth(self.coods, "stench.png")
                for i in valid_coods:
                       if i not in self.safe_squares:
                            self.rates_table[i]-=10
                break
        if safePit and safeWumpus:
            for _ in valid_coods:
                            self.rates_table[valid_coods]+=0 
    
    
    def episode(self):
        self.p=8
        self.greedy = 1
        self.random = 0
        self.print_text(' Episode:{}'.format("test"),(180,50),30)
        self.initial_state()


        
        if self.to_cood(self.coods)==self.gold_state:
            self.initial_state()
            clock.tick(100)
            pygame.display.update()
            pygame.quit()
            

            raise Exception('game ended')

        self.rates_table[self.to_cood(self.coods)]-=10
        if self.game_list[self.to_cood(self.coods)]=="square.png" and self.to_cood(self.coods)!=self.wump_state:   
            self.safe_squares.append(self.to_cood(self.coods))
        else:
            self.rates_table[self.to_cood(self.coods)]-=1000
            self.coods=self.initial_cood
        
        valid_moves=self.valid_moves(self.coods)
        self.initial_state()
        valid_coods = [self.to_cood((int(self.coods[0] + self.action_space[i]['x']), int(self.coods[1] + self.action_space[i]['y']))) for i in valid_moves]
        trapped=True
        if(self.to_cood(self.coods) not in self.already_visited):
              
            self.check_moves(valid_coods)
            self.already_visited.append(self.to_cood(self.coods))
        else:
            for v_c in valid_coods:
                  if self.game_list[v_c]=="square.png":
                       trapped=False
            if trapped:
                clock.tick(1000)
                pygame.quit()
                raise Exception('No puedo salir')
                 
        #episodio de training
        

        pygame.event.get()
        rates=[self.rates_table[j] for j in valid_coods]

        if len(rates)==len(set(rates)):

            new_moves=[valid_moves[i] for i in range(len(valid_coods)) if self.rates_table[valid_coods[i]]== max(rates)]
        else: 
            rate_mode, _ = (Counter(rates).most_common(1)[0])
            new_moves=[valid_moves[i] for i in range(len(valid_coods)) if self.rates_table[valid_coods[i]]==rate_mode]

        moves_weights=[1/len(new_moves) for _ in new_moves]
        action = random.choices(new_moves,weights=moves_weights,k=1)
        action=action[0]
        self.coods=((int(self.coods[0] + self.action_space[action]['x']), int(self.coods[1] + self.action_space[action]['y'])))
        pygame.display.update()
        clock.tick(10)


         

    def print_text(self,text,cood,size):
        font = pygame.font.Font(pygame.font.get_default_font(), size)
        text_surface = font.render(text, True, (255,255,255))
        DISPLAYSURF.blit(text_surface,cood)

    def valid_moves(self,cood):
        actions=[]      
        for i in range(4):
                new_cood = (int(cood[0] + self.action_space[i]['x']), int(cood[1] + self.action_space[i]['y']))
                if self.is_valid_move(new_cood):
                      actions.append(i)
                      #actions.append(self.to_cood(new_cood))
        return actions               

    def training(self, epoch):

                    self.p=4800
                    self.print_text('Moves:{}'.format(epoch),(180,50),30)
                    self.episode()             
                    clock.tick(1)


            
    def testing(self,initial_state=0):
            self.p=8
            self.greedy = 1
            self.random = 0
            self.print_text(' Episode:{}'.format("test"),(180,50),30)
            self.initial_state()
            self.episode(initial_state,True,self.p)
            clock.tick(8)
  
 

print("Start training?: (y/n)")
desicion=input()
if desicion.upper()!="N":
    game = game_env(4)
    pygame.init()
    count=1
    flag = 0
    DISPLAYSURF = pygame.display.set_mode(game.game_dim,0,32)
    clock = pygame.time.Clock()
    while True: 
        game.episode()
        for event in pygame.event.get():
                    if event.type==QUIT:
                                pygame.quit()
                                flag = 1
                                raise Exception('game ended')
        if flag==1:
            break
 

flag = 0
pygame.init()
pygame.display.set_caption('Reinforcement Learning')
DISPLAYSURF = pygame.display.set_mode(game.game_dim,0,32)
clock = pygame.time.Clock()