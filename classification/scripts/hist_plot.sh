## PS x SM

topcat -stilts plot2plane \
   xpix=600 ypix=580 \
   ylog=true xlabel='P_{STAR}^{DESW} - P_{STAR}^{PSW}' ylabel='Number\:of\:sources' texttype=latex fontsize=34 fontstyle=serif \
   xmin=-1 xmax=1 ymin=500 ymax=1e7 \
   legend=false \
   layer=Histogram \
      in='../DESxPS_1.5.fits' \
      x=preds0_1-preds0_2 \
      color=black binsize=-55 barform=semi_steps  out="DES_PS_STAR.png"

topcat -stilts plot2plane \
   xpix=600 ypix=580 \
   ylog=true xlabel='P_{QSO}^{DESW} - P_{QSO}^{PSW}' ylabel='Number\:of\:sources' texttype=latex fontsize=34 fontstyle=serif \
   xmin=-1 xmax=1 ymin=300 ymax=1e7 \
   legend=false \
   layer=Histogram \
      in='../DESxPS_1.5.fits' \
      x=preds1_1-preds1_2 \
      color=black binsize=-55 barform=semi_steps out="DES_PS_QSO.png"


topcat -stilts plot2plane \
   xpix=600 ypix=580 \
   ylog=true xlabel='P_{GALAXY}^{DESW} - P_{GALAXY}^{PSW}' ylabel='Number\:of\:sources' texttype=latex fontsize=34 fontstyle=serif \
   xmin=-1 xmax=1 ymin=1000 ymax=1e7 \
   legend=false \
   layer=Histogram \
      in='../DESxPS_1.5.fits' \
      x=preds2_1-preds2_2 \
      color=black binsize=-55 barform=semi_steps  out="DES_PS_GALAXY.png"



## PS x SM

topcat -stilts plot2plane \
   xpix=600 ypix=580 \
   ylog=true xlabel='P_{STAR}^{PSW} - P_{STAR}^{SMW}' ylabel='Number\:of\:sources' texttype=latex fontsize=34 fontstyle=serif \
   xmin=-1 xmax=1 ymin=500 ymax=1e7 \
   legend=false \
   layer=Histogram \
      in='../PSxSM_1.5.fits' \
      x=preds0_1-preds0_2 \
      color=black binsize=-55 barform=semi_steps  out="PS_SM_STAR.png"

topcat -stilts plot2plane \
   xpix=600 ypix=580 \
   ylog=true xlabel='P_{QSO}^{PSW} - P_{QSO}^{SMW}' ylabel='Number\:of\:sources' texttype=latex fontsize=34 fontstyle=serif \
   xmin=-1 xmax=1 ymin=10 ymax=1e7 \
   legend=false \
   layer=Histogram \
      in='../PSxSM_1.5.fits' \
      x=preds1_1-preds1_2 \
      color=black binsize=-55 barform=semi_steps out="PS_SM_QSO.png"


topcat -stilts plot2plane \
   xpix=600 ypix=580 \
   ylog=true xlabel='P_{GALAXY}^{PSW} - P_{GALAXY}^{SMW}' ylabel='Number\:of\:sources' texttype=latex fontsize=34 fontstyle=serif \
   xmin=-1 xmax=1 ymin=500 ymax=1e7 \
   legend=false \
   layer=Histogram \
      in='../PSxSM_1.5.fits' \
      x=preds2_1-preds2_2 \
      color=black binsize=-55 barform=semi_steps  out="PS_SM_GALAXY.png"




## SM x DES

topcat -stilts plot2plane \
   xpix=600 ypix=580 \
   ylog=true xlabel='P_{STAR}^{SMW} - P_{STAR}^{DESW}' ylabel='Number\:of\:sources' texttype=latex fontsize=34 fontstyle=serif \
   xmin=-1 xmax=1 ymin=1000 ymax=1e7 \
   legend=false \
   layer=Histogram \
      in='../SMxDES_1.5.fits' \
      x=preds0_1-preds0_2 \
      color=black binsize=-55 barform=semi_steps  out="SM_DES_STAR.png"

topcat -stilts plot2plane \
   xpix=600 ypix=580 \
   ylog=true xlabel='P_{QSO}^{SMW} - P_{QSO}^{DESW}' ylabel='Number\:of\:sources' texttype=latex fontsize=34 fontstyle=serif \
   xmin=-1 xmax=1 ymin=100 ymax=1e7 \
   legend=false \
   layer=Histogram \
      in='../SMxDES_1.5.fits' \
      x=preds1_1-preds1_2 \
      color=black binsize=-55 barform=semi_steps out="SM_DES_QSO.png"


topcat -stilts plot2plane \
   xpix=600 ypix=580 \
   ylog=true xlabel='P_{GALAXY}^{SMW} - P_{GALAXY}^{DESW}' ylabel='Number\:of\:sources' texttype=latex fontsize=34 fontstyle=serif \
   xmin=-1 xmax=1 ymin=1000 ymax=1e7 \
   legend=false \
   layer=Histogram \
      in='../SMxDES_1.5.fits' \
      x=preds2_1-preds2_2 \
      color=black binsize=-55 barform=semi_steps  out="SM_DES_GALAXY.png"



