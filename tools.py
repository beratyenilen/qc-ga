import pickle
from datetime import datetime

#   Functions for handling and analyzing 
#   the population and logbook data


#   Save a population object and a logbook   
def save(pop, logbook, path, state_name="unknown_state"):
    n = pop[0].numberOfQubits
    NGEN = len(logbook.select("gen")) 
    time = datetime.now()
    time_str = time.strftime("%d.%m.%y-%H:%M")
    ID = time.strftime("%d%m%y%H%M%S")+str(len(pop))+str(NGEN)+str(n)   #This needs improving
    f = open(path+ID+"-"+time_str+"-"+str(len(pop))+"pop-"+str(NGEN)+"GEN-"+state_name+".pop", 'wb')
    pickle.dump(pop, f)
    f.close()
    f = open(path+ID+"-"+time_str+"-"+str(len(pop))+"pop-"+str(NGEN)+"GEN-"+state_name+".logbook", 'wb')
    pickle.dump(logbook, f)
    f.close()
    print('Saved!')

#
#   Load a population object and corresponding logbook.
#
#   Path contains the name of pop/logbook file 
#   WITHOUT .pop/.logbook extension
#
def load(path):
    f = open(path+".pop", 'rb')
    pop = pickle.load(f)
    f.close()
    f = open(path+".logbook", 'rb')
    logbook = pickle.load(f)
    f.close()
    return pop, logbook
    
#   Load allowed gateset, fitness and seed from a file
def getSetup(path):
    return 0
