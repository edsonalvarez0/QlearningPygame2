import pygame, sys
from pygame.locals import *
import random
import numpy as np
class game_env:
    def __init__(self,size):
        self.alfa = 0.75
        self.beta = 0.75
        self.greedy = 0
        self.random = 1
        self.delta = 0.005
        self.q_table = np.zeros((size**2,4))
        self.reward_map = {'square.png':-10,'pit.png':-1000,'gold.png':1000,'already_visited':-10,'invalid':-1000}
        self.game_dim = (600,600)
        self.first_cell=0
        self.last_cell=size**2 - 1
        self.text_space=0
        self.initial_cood = (0,0+ self.text_space)
        self.rows,self.columns= size,size
        self.cell_dim = self.game_dim[0]/self.rows
        self.final_cood = (self.game_dim[0]-self.cell_dim, self.game_dim[1]-self.cell_dim)
        self.game_grid = self.new_game_env()

        self.action_space = {0:{'x':-1*self.cell_dim,'y':0},
                              2:{'x':self.cell_dim,'y':0}, 
                              1:{'x':0,'y':self.cell_dim},
                              3:{'x':0,'y':-1*self.cell_dim}}



    def new_game_env(self):
        #Crea el tablero
        matrix = random.choices(['square.png','pit.png'],weights=[0.8,0.2], k=self.rows*self.columns)
        matrix = np.asarray(matrix).reshape(self.rows,self.columns)
        matrix[0][0] ='square.png'
        self.gold_state=random.randint(self.first_cell,self.last_cell)
        matrix[self.gold_state%self.rows][self.gold_state//self.rows] = 'gold.png'
        return matrix    

    def load_images(self,img_path):
        img = pygame.image.load('{}'.format(img_path))
        img = pygame.transform.scale(img,(self.cell_dim,self.cell_dim))
        return img  
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
                    img = self.load_images(self.game_grid[x][y])
                    cood = (y*self.cell_dim,x*self.cell_dim)
                    DISPLAYSURF.blit(img,cood)
        pygame.display.update()
    def steps_visualizer(self,cood):
        img = pygame.image.load('agent .png')
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
            if self.game_grid[self.to_cood(cood)//self.columns][self.to_cood(cood)%self.rows]=="pit.png":
                return False
            else:
                return True         
        return False        
    

    def valid_moves(self,cood):
        actions=[]      
        for i in range(4):
                new_cood = (int(cood[0] + self.action_space[i]['x']), int(cood[1] + self.action_space[i]['y']))
                if self.initial_cood[0]<=new_cood[0]<=self.final_cood[0] and self.initial_cood[1]<=new_cood[1]<=self.final_cood[1]:
                      actions.append(i)

        return actions    
    
    def check_moves(self,curr_cood,valid_coods,valid_actions,already_visited): #v_coods in self.to_pygame_cood
        safePit=True
        for v_c in valid_coods:
            if self.game_grid[v_c//self.rows][v_c%self.rows]=="pit.png" and safePit:
                  safePit=False
                  if self.display:
                    self.display_smth(curr_cood, "breeze.png")
                  for i in valid_actions:
                       if (int(curr_cood[0] + self.action_space[i]['x']), int(curr_cood[1] + self.action_space[i]['y'])) not in already_visited:
                            self.q_table[self.to_cood(curr_cood)][i]-=10
                  break



    def q_table_update(self,  state, action, already_visited):
        

        curr_cood = self.to_pygame_cood(state)
        new_cood = (int(curr_cood[0] + self.action_space[action]['x']), int(curr_cood[1] + self.action_space[action]['y']))
        new_state = self.to_cood(new_cood)
        is_valid = self.is_valid_move(new_cood, already_visited)     
        valid_actions=self.valid_moves(curr_cood)
        valid_coods = [self.to_cood((int(curr_cood[0] + self.action_space[i]['x']), int(curr_cood[1] + self.action_space[i]['y']))) for i in valid_actions]
        trapped=True
        for v_c in valid_coods:
            if self.game_grid[v_c//self.rows][v_c%self.rows]=="square.png":
                trapped=False
        if trapped:
            clock.tick(1000)
            pygame.quit()
            raise Exception('No puedo salir')
        self.check_moves(curr_cood,valid_coods,valid_actions,already_visited)
  
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
        count=0
        cood = self.to_pygame_cood(current_state)
        already_visited = [cood]
        self.steps_visualizer(cood)
        while is_valid==True:
            pygame.display.set_caption('Moves={}'.format(count))
            pygame.display.update()
            for event in pygame.event.get():
                if event.type==QUIT:
                                pygame.quit()
                                raise Exception('Game Ended')
            choice = random.choices([True,False],weights=[self.greedy,self.random],k=1)
            if choice[0]:
                     action = np.argmax(self.q_table[current_state])
            else:
                     action = random.choices([0,1,2,3],weights=[0.25,0.25,0.25,0.25],k=1)
                     action = action[0]
            is_valid, current_state, cood = self.q_table_update(current_state, action, already_visited)
            count+=1        
            
            pygame.display.update()
            clock.tick(p)
            already_visited.append(cood)
            if is_valid:
                self.steps_visualizer(cood)
                lastcell=(self.to_cood(cood))
        

        return (self.game_grid[lastcell//self.columns][lastcell%self.rows]=="gold.png")
 
    def episode_no_sprites(self, current_state, is_valid):
        #episodio de training
        cood = self.to_pygame_cood(current_state)

        already_visited = [cood]     
        while is_valid==True:
            choice = random.choices([True,False],weights=[self.greedy,self.random],k=1)
            if choice[0]:
                     action = np.argmax(self.q_table[current_state])
            else:
                     action = random.choices([0,1,2,3],weights=[0.25,0.25,0.25,0.25],k=1)
                     action = action[0]
            is_valid, current_state, cood = self.q_table_update(current_state, action, already_visited)           
            already_visited.append(cood)       

    def training_no_sprites(self, epoch):
                        self.display=False
                        self.p=8
                        state=self.first_cell
                        self.episode_no_sprites(state, True)  
                        #print('episode {} ---->'.format(epoch))
                        if epoch%50==0:
                            if self.random>0:
                                    self.greedy+=self.delta
                                    self.random-=self.delta
                                    self.greedy = min(self.greedy,1)
                                    self.random= max(self.random,0)                      
                        if epoch%(200)==0:
                            self.delta*=2
                        if epoch%1000==0:
                              dummy_greedy=self.greedy
                              dummy_random=self.random
                              finished=self.testing()
                              if finished:
                                    return finished
                              else:
                                    self.greedy = dummy_greedy
                                    self.random = dummy_random
          
                        return False

            
    def testing(self,initial_state=0):
            self.display=True
            self.p=8
            self.greedy = 1
            self.random = 0
            self.initial_state()
            finsihed=self.episode(initial_state,True,self.p)
            clock.tick(2)
            return finsihed          

print("Start training?: (y/n)")
desicion=input()
if desicion.upper()!="N":
    size=4
    game = game_env(size)
    pygame.init()
    count=1
    flag = 0
    DISPLAYSURF = pygame.display.set_mode(game.game_dim,0,32)
    clock = pygame.time.Clock()
    while True:
        count+=1
        finishes=game.training_no_sprites(count)
        if finishes:
            pygame.quit()
            break              
        if count==3000:
            pygame.quit()
            break
        for event in pygame.event.get():
                    if event.type==QUIT:
                                pygame.quit()
                                flag = 1
                                raise Exception('game ended')
        if flag==1:
            break

flag = 0
pygame.init()
pygame.display.set_caption('Wumpus Reinforcement Learning')
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