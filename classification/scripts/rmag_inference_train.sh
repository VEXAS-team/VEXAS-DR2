stilts plot2plane \
   xpix=700 ypix=580 \
   ylog=true xlabel='r,\:[mag]' ylabel= grid=true texttype=latex fontsize=30 fontstyle=serif \
   xmin=13 xmax=25 ymin=1 ymax=9000000 \
   legend=true legpos=0.0,1.0 \
   x=mag_auto_r_DES binsize=-218 barform=steps \
   layer_1=Histogram \
      in_1=/mnt/DATA/VLAD/VEXAS/data/final_tables/DES_final.fits \
      color_1=9999ff \
      leglabel_1=DESW \
   layer_2=Histogram \
      in_2=/mnt/DATA/VLAD/VEXAS/repo/vexas_dr2/_VEXAS-DR2/classification/data/train/DES_SPEC_ALL.csv ifmt_2=CSV \
      color_2=ff0033 transparency_2=1.0E-4 \
      leglabel_2='DESW\:training' 
