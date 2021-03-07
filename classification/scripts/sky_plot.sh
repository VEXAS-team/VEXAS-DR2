survey=DES

# STAR
topcat -stilts plot2sky \
   xpix=700 ypix=500 \
   labelpos=None crowd=1.22 texttype=latex fontsize=34 fontstyle=serif \
   clon=18 clat=-27.9 radius=90 \
   auxmap=rdbu auxclip=0,1 auxfunc=sqrt auxmin=10 \
   auxvisible=true auxlabel='Sources\;per\;deg^2' \
   title=STAR legend=false \
   layer=SkyDensity \
      in="/mnt/DATA/VLAD/VEXAS/data/"$survey"_result.fits" icmd='select "source_class== 0 && preds0 > 0.7"' \
      lon=ra lat=dec \
      level=-1 out="./sky/"$survey"_STAR.png"


# QSO
topcat -stilts plot2sky \
   xpix=700 ypix=500 \
   labelpos=None crowd=1.22 texttype=latex fontsize=34 fontstyle=serif \
   clon=18 clat=-27.9 radius=90 \
   auxmap=rdbu auxclip=0,1 auxfunc=sqrt auxmin=1 auxmax=400 \
   auxvisible=true auxlabel='Sources\;per\;deg^2' \
   title=QSO legend=false \
   layer=SkyDensity \
      in="/mnt/DATA/VLAD/VEXAS/data/"$survey"_result.fits" icmd='select "source_class== 1 && preds1 > 0.7"' \
      lon=ra lat=dec \
      level=-1 out="./sky/"$survey"_QSO.png"

# GALAXY
topcat -stilts plot2sky \
   xpix=700 ypix=500 \
   labelpos=None crowd=1.22 texttype=latex fontsize=34 fontstyle=serif \
   clon=18 clat=-27.9 radius=90 \
   auxmap=rdbu auxclip=0,1 auxfunc=sqrt auxmin=10 \
   auxvisible=true auxlabel='Sources\;per\;deg^2' \
   title=GALAXY legend=false \
   layer=SkyDensity \
      in="/mnt/DATA/VLAD/VEXAS/data/"$survey"_result.fits" icmd='select "source_class== 2 && preds2 > 0.7"' \
      lon=ra lat=dec \
      level=-1 out="./sky/"$survey"_GALAXY.png"


survey=PS

# STAR
topcat -stilts plot2sky \
   xpix=700 ypix=500 \
   labelpos=None crowd=1.22 texttype=latex fontsize=34 fontstyle=serif \
   clon=18 clat=-27.9 radius=90 \
   auxmap=rdbu auxclip=0,1 auxfunc=sqrt auxmin=10 \
   auxvisible=true auxlabel='Sources\;per\;deg^2' \
   title=STAR legend=false \
   layer=SkyDensity \
      in="/mnt/DATA/VLAD/VEXAS/data/"$survey"_result.fits" icmd='select "source_class== 0 && preds0 > 0.7"' \
      lon=ra lat=dec \
      level=-1 out="./sky/"$survey"_STAR.png"


# QSO
topcat -stilts plot2sky \
   xpix=700 ypix=500 \
   labelpos=None crowd=1.22 texttype=latex fontsize=34 fontstyle=serif \
   clon=18 clat=-27.9 radius=90 \
   auxmap=rdbu auxclip=0,1 auxfunc=sqrt auxmin=1 \
   auxvisible=true auxlabel='Sources\;per\;deg^2' \
   title=QSO legend=false \
   layer=SkyDensity \
      in="/mnt/DATA/VLAD/VEXAS/data/"$survey"_result.fits" icmd='select "source_class== 1 && preds1 > 0.7"' \
      lon=ra lat=dec \
      level=-1 out="./sky/"$survey"_QSO.png"

# GALAXY
topcat -stilts plot2sky \
   xpix=700 ypix=500 \
   labelpos=None crowd=1.22 texttype=latex fontsize=34 fontstyle=serif \
   clon=18 clat=-27.9 radius=90 \
   auxmap=rdbu auxclip=0,1 auxfunc=sqrt auxmin=10 \
   auxvisible=true auxlabel='Sources\;per\;deg^2' \
   title=GALAXY legend=false \
   layer=SkyDensity \
      in="/mnt/DATA/VLAD/VEXAS/data/"$survey"_result.fits" icmd='select "source_class== 2 && preds2 > 0.7"' \
      lon=ra lat=dec \
      level=-1 out="./sky/"$survey"_GALAXY.png"


