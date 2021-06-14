import pickle
from datetime import datetime

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


