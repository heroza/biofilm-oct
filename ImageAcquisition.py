#acquire image in the current position
def makeScan(stp_position='default',params={}):
    #OctControl.setParameters(**params)
    params['colorBoundaries']=[20.0,90.0]
    #OctControl.setParameters(colorBoundaries=[20.0,80.0])
    params['path']="../Measurments/%Y-%m-%d_%H-%M-%S_"+stp_position
    #OctControl.setParameters(path="../Measurments/%Y-%m-%d"+stp_position)
    params['exportRAWGrayscale8bit']="%Y-%m-%d_%H-%M-%S_"+stp_position
    #OctControl.setParameters(exportRAWGrayscale8bit="%Y-%m-%d"+stp_position)
    params['exportSRM']="%Y-%m-%d_%H-%M-%S_"+stp_position
    #OctControl.setParameters(exportSRM="%Y-%m-%d"+stp_position)    
    #OctControl.setParameters(path="../Measurments/%Y-%m-%d/%Y-%m-%dbla_"+stp_position)
    #OctControl.setParameters(exportRAWGrayscale8bit="%Y-%m-%d_"+stp_position)
    #OctControl.setParameters(exportSRM="%Y-%m-%d_"+stp_position)
    try:
        stp.actualizeStatus()
        #OctControl.setParameters(stepcraftStatus=stp.status)
        params['stepcraftStatus']=stp.status
    except:
        print('cannot set stepcraft status')
        
    meta=OctControl.scan(**params)
    
    with open(meta['path_raw8bit'].replace('.raw','.json'), 'w') as fp:
        json.dump(meta, fp)
    time.sleep(5) 

#acquire image in position defined in pos
def makeMoveScan(pos=None,stp_position='default',params={}):

    if pos is None:
        pos=positions[stp_position]
    stp.moveTo(z=-0.5)
    stp.moveTo(x=pos[0],y=pos[1])
    stp.moveTo(z=pos[2])
    #print('scan '+stp_position+' @'+str(pos))
    stp.waitFinish()
    makeScan(stp_position,params=params)
    stp.moveTo(z=-0.5)

    
#acquire images in tiling array around position "stp_position", defined in the dictionary "positions"
def multipleScan(stp_position,n=(1,1),params={}):
         
    pos=positions[stp_position]
    offset=7.0#(-8.0,0,8.0)   #set here the overlap percentage between neighboring images (1-0.7=30% here)
    
    for Yscn in range(n[1]):
        for Xscn in range(n[0]):
            #for o in offset
            
            xoffs=((-n[0]+1)/2+Xscn)*offset+pos[0]
            yoffs=((-n[1]+1)/2+Yscn)*offset+pos[1]
            pOffs=(xoffs,yoffs,pos[2])
            #pOffs=(xoffs,yoffs,(pos[2]+0.802))    #offset here
            stp_positionFull=stp_position+'-'+str(Xscn+Yscn*n[1])
           # print (pOffs)
            makeMoveScan(pos=pOffs,stp_position=stp_positionFull,
                         params=params)

#time execution
def tic():
    import time
    global startTime_for_tictoc
    startTime_for_tictoc = time.time()

def toc():
    import time
    if 'startTime_for_tictoc' in globals():
        t=time.time();
        print("Elapsed time is " + str(time.time() - startTime_for_tictoc) + " seconds.")  
    else:
        print("Toc: start time not set")