import warnings
warnings.filterwarnings("ignore")

# FUNCTION ADOPTED FROM CJH LUDWIG, JANUARY 2002
# This function has a matrix as input that contains four columns: blocknumber, trialnumber, x and y starting positions. 
# It goes through this matrix, normalising each movement and fitting a quadratic, and cubic polynomial on the samples.
# Several metrics of curvature are calculated: Initial deviation (Van Gisbergen et al. 1987), average initial deviation (Sheliga
# et al., 1995), maximum raw deviation (Smit et al., 1990; Doyle and Walker, 2001,2002), and an area-based measure (Doyle and Walker,
# pers. communication). In addition to these existing metrics, we calculate two metrics derived from the curve fitting procedure 


# written by Paul Prasse

import numpy as np


def logical_xor(a, b):
    if bool(a) == bool(b):
        return False
    else:
        return a or b

# function that returns curve metrics for given x- and y- coordinates
# usually called with the x- and y- coordinates of a saccade
#
# params:
#   y_dva: degrees of visual angle (y-axis)
#   x_dva: degrees of visual angle (x-axis)
#   sampling_rate: sampling rate 
def curve_metrics(x_dva,y_dva, sampling_rate):

    metrics = dict()
    x=x_dva
    y=y_dva
    xnorm=[]
    ynorm=[]
   
    if x[-1]>x[0]:
        direction=1 #rightward saccade
    else:
        direction=0 #left saccade
      
    NRsamples=len(x) #number of samples on this trial

    hordisplacement=x[-1]-x[0]
    vertdisplacement=y[0]-y[-1]
    Hstraight=np.sqrt((hordisplacement**2)+(vertdisplacement**2))
    SacAngle=np.arctan2(vertdisplacement,hordisplacement)*(180/np.pi) #calculate the angle of the entire movement
    
    #build up the normalised vectors
    xnorm.append(0) #each movement is normalised so that the starting position coincides with this origin (0,0)
    ynorm.append(0)
    xres=[]
    
    for SampleIndex in np.arange(1,(NRsamples-1),1): #first and last samples never deviate from the straight trajectory!
        hordisplacement= x[SampleIndex]-x[0]
        vertdisplacement= y[0]-y[SampleIndex]
        Hsample=np.sqrt((hordisplacement**2)+(vertdisplacement**2))
        SamAngle=np.arctan2(vertdisplacement,hordisplacement)*180/np.pi
        if SacAngle>SamAngle:
            devdir=1 #clockwise deviation
            DevAngle=SacAngle-SamAngle
        elif SacAngle<SamAngle:
            devdir=-1 #anti-clockwise deviation
            DevAngle=SamAngle-SacAngle
        else:
            devdir=0 #no deviation
            DevAngle=0
            
        Deviation=np.sin(DevAngle*(np.pi/180))*Hsample
        Deviation=Deviation*devdir
        xtrue=np.sqrt((Hsample**2)-(Deviation**2)) #true x-coordinate along the straight line path
        xnorm.append(xtrue)
        ynorm.append(Deviation)
    
    xnorm.append(Hstraight)
    ynorm.append(0)
    # rescale the x-coordinates so that xstart=-1 and xend=1
    for SampleIndex in np.arange(0,NRsamples,1):
        res=-1+((xnorm[SampleIndex]/xnorm[-1])*2)
        xres.append(res)
    
    #    %now calculate the various established curvature metrics
    #    IniDev=atan2(ynorm(3),xnorm(3))*180/pi; %initial deviation metric of Van Gisbergen et al. (1987)
    #    IniAD=mean(ynorm(2:3)); %initial average deviation of Sheliga et al. (1995)
    cPoint = int(np.round(0.005*sampling_rate)) # 5ms
    if cPoint < len(xnorm) or cPoint < 1:
        IniDev=0
        IniAD=0
    if logical_xor(cPoint < len(ynorm),cPoint < 1):
        IniDev=np.arctan2(ynorm[cPoint],xnorm[cPoint])*180/np.pi #initial deviation metric of Van Gisbergen et al. (1987)
        IniAD=np.mean(ynorm[1:cPoint]) #initial average deviation of Sheliga et al. (1995)
    else:
       IniDev=np.arctan2(ynorm[-1],xnorm[-1])*180/np.pi #initial deviation metric of Van Gisbergen et al. (1987)
       IniAD=np.mean(ynorm[1:]) #initial average deviation of Sheliga et al. (1995)
    
    MaxDev = np.max(ynorm)
    MaxIndex = np.argmax(ynorm) #maximum raw deviation (Smit et al., 1990; Doyle and Walker, 2001,2002)
    MinDev = np.min(ynorm)
    MinIndex = np.argmin(ynorm)
    
    if np.abs(MaxDev) > np.abs(MinDev):
        RawDev=MaxDev
        DevIndex=MaxIndex
    elif np.abs(MaxDev) < np.abs(MinDev):
        RawDev=MinDev
        DevIndex=MinIndex
    else:
        if MaxIndex<MinIndex:
            RawDev=MaxDev
            DevIndex=MaxIndex
        else:
            RawDev=MinDev
            DevIndex=MinIndex
            
    RawDev=(RawDev/xnorm[-1])*100
    RawPOC=(xnorm[DevIndex]/xnorm[-1])*100 #raw point of curvature
   
    AreaVector=[] #area based measure (Doyle and Walker, personal communication)
    AreaVector.append(0)
    for AreaIndex in np.arange(1,len(xnorm),1):
        area=(xnorm[AreaIndex]-xnorm[AreaIndex-1])*(ynorm[AreaIndex-1]/2)
        AreaVector.append(area)
   
    CurveArea=(np.sum(AreaVector)/xnorm[-1])*100;
    
    #fit the quadratic function and determine the direction of curvature
    """
    print('############# values: #############')
    print(x)
    print(xres)
    print(y)
    print(ynorm)
    print()
    print()
    """
    pol2 = np.polyfit(xres, ynorm, 2)
    polyval = np.poly1d(pol2)
    ypred2=polyval(xres)
    if pol2[0]<0: #if quadratic coefficient is negative (upward curve), curvature is clockwise
        pol2[0] = np.abs(pol2[0])
    else:
        pol2[0] = pol2[0] * -1 #if quadratic coefficient is positive (downward curve), curvature is anti-clockwise
    
    pol3 = np.polyfit(xres, ynorm, 3) #%derivative of cubic polynomial
    polyval3 = np.poly1d(pol3)
    ypred3=polyval3(xres)
    
    vertdisplacement=ypred3[0]-ypred3[-1]
    Hstraight=np.sqrt((xnorm[-1]**2)+(vertdisplacement**2))
    SacAngle=np.arctan2(vertdisplacement,xnorm[-1])*(180/np.pi)
    
    
    der3=np.polyder(pol3) #derivative of cubic polynomial gives a maximum and minimum
    xder3=[((-1*der3[1])-np.sqrt((der3[1]**2)-(4*der3[0]*der3[2])))/(2*der3[0])]
    xder3.append(((-1*der3[1])+np.sqrt((der3[1]**2)-(4*der3[0]*der3[2])))/(2*der3[0]))
    if ((xder3[0]<xres[0]) or (xder3[0]>xres[-1])): #check whether first maximum/minimum falls within the range of xres
        curve3=0
        POC3=0
    elif not np.all(np.isreal(xder3)):
        curve3=0
        POC3=0
    else:   #%if yes, then calculate curvature
        ymax3=polyval3(xder3[0])
        POC3=(xder3[0]*np.std(xnorm))+np.mean(xnorm)
        POC3=(POC3/xnorm[-1]*100)
        hordisplacement=(xder3[0]*np.std(xnorm))+np.mean(xnorm)
        vertdisplacement= ypred3[0]-ymax3
        Hsample=np.sqrt((hordisplacement**2)+(vertdisplacement**2))
        SamAngle=np.arctan2(vertdisplacement,hordisplacement)*180/np.pi
        if  SacAngle>SamAngle:
            devdir=1
            DevAngle=SacAngle-SamAngle
        elif SacAngle<SamAngle:
            devdir=-1
            DevAngle=SamAngle-SacAngle
        curve3=np.sin(DevAngle*(np.pi/180))*Hsample
        curve3=((curve3*devdir)/xnorm[-1])*100  
    
    # create list
    curve3 = [curve3]
    POC3   = [POC3]
    if ((xder3[1]<xres[0]) or (xder3[1]>xres[-1])): #check whether second maximum/minimum falls within the range of xres
        curve3.append(0)
        POC3.append(0)
    elif not np.all(np.isreal(xder3)):           
        curve3.append(0)
        POC3.append(0)
    else:
        ymax3=polyval3(xder3[1]);
        POC=(xder3[1]*np.std(xnorm))+np.mean(xnorm)
        POC3.append(POC/xnorm[-1]*100)
        hordisplacement=(xder3[1]*np.std(xnorm))+np.mean(xnorm)
        vertdisplacement = ypred3[0]-ymax3
        Hsample=np.sqrt((hordisplacement**2)+(vertdisplacement**2))
        SamAngle=np.arctan2(vertdisplacement,hordisplacement)*180/np.pi
        if SacAngle>SamAngle:
            devdir=1#clockwise deviation
            DevAngle=SacAngle-SamAngle
        elif SacAngle<SamAngle:
            devdir=-1#anti-clockwise deviation
            DevAngle=SamAngle-SacAngle
        curve=np.sin(DevAngle*(np.pi/180))*Hsample
        curve3.append(((curve*devdir)/xnorm[-1])*100)      
    if np.max(np.abs(curve3))> 0:
        MaxDev = np.max(np.abs(curve3))
        MaxIndex = np.argmax(np.abs(curve3))
    else:
        MaxIndex=1
    
    metrics['direction'] = direction
    metrics['IniDev'] = IniDev
    metrics['IniAD'] = IniAD
    metrics['RawDev'] = RawDev
    metrics['RawPOC'] = RawPOC
    metrics['CurveArea'] = CurveArea
    metrics['pol2[0]'] = pol2[0]
    metrics['curve3[0]'] = curve3[0]
    metrics['POC3[0]'] = POC3[0]
    metrics['curve3[1]'] = curve3[1]
    metrics['POC3[1]'] = POC3[1]
    metrics['curve3[MaxIndex]'] = curve3[MaxIndex]
    metrics['POC3[MaxIndex]'] = POC3[MaxIndex]
    return metrics