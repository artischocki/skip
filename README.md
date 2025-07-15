# Doku um die Präsi zu erstellen

## 0. Prerequisites
Falls ihr selber nochmal die ganzen Bilder die ich nicht gepusht habe
generieren wollt, oder selber evaluieren oder so, benötigt ihr:
1. Python ^= 3.10
2. poetry

Poetry venv installieren mit:
```
poetry install
```
Danach files ausführen mit:
```
poetry run python pipeline/train/preprocess.py # Beispiel
```

In den Notebooks steht mittlerweile nichts sinnvolles mehr drin, die habe ich
nur zum rumexperimentieren benutzt.

Alles was für uns relevant ist steht mittlerweile in `/pipeline/`.


## 1. Data Preprocessing
```
pipeline/train/preprocess.py
```
1. Wir hatten noch die zusätzlichen Bilder und Labels aus dem kip2 repo, die
   ich versehentlich gelabelt habe. Insgesamt hatten wir also 60 Bilder +
   Labels. Ich habe diese unterteilt in train/val/split, siehe
   `data/training/*/raw`. Train: 70% (42), Val: 20% (12), Test: 10% (6)

2. Dann das Preprocessing wie in den Notebooks beschrieben:
    1. ROI cropping: -> `data/training/*/cropped`
    2. Resizing: -> `data/training/*/resized`
    3. Augmentierung -> `data/training/*/augmented`

3. Zur Augmentierung: Ich habe da für jedes Bild folgendes vorgenommen:
- Rotate: -> jedes der Bilder wird jeweils um die Winkel [-3, -2, -1, 0, 1, 2, 3] gedreht
    - Für jedes dieser gedrehten Bilder mache ich folgende augmentierung:
        1. Nichts
        2. H-Flip
        3. V-Flip
        4. H-Flip + V-Flip (entsprivht Drehung um 180Grad)
- Es ergibt sich ein datensatz von 1680 Bildern (summiert aus train/val/test)
- Ich habe auch nur rotations trainiert. (siehe 2. Training)

Wir haben auch mit Noise, Kontrastverstellung und Gaussian Blur rumprobiert,
aber haben diese Modelle wieder verworfen, weil sie am Ende schlechter performt
haben auf den Test daten. Ist auch irgendwie logisch, weil das Netz lernt
generalisieren aber verschwommene Bilder oder körnige Bilder ähneln den Test
daten überhaupt nicht. Die Generalisierung die er beim Flippen und Rotieren
lernt sind ausreichend.

Ich habe auch zusätzlich overlays generiert, da sieht man die labels, die kann
man evtl auch für die präsi verwenden. Pushe die auch.



## 2. Training
```
pipeline/train/train.py
```
Insgesamt wurden bestimmt über 10 Netze trainiert aber ich schlage vor für die
Präsi beschränken wir uns auf den finalen run mit `correct_split_only_rotations`
und `correct_split_all_augmentations`.

- Hier haben wir uns an die im Notebook vorgegebene Netz-Architektur gehalten.
- Im Training zusätzlich Early Stopping und Checkpointing implementiert
- Training des finalen Netzes: 20 Epochen, je ca. 30min -> 10h (nur cpu)
- Hier sind die 2 loss plots für 1. correct_split_all_augmentations und 2.
  correct_split_only_rotation:
![Loss Plot 1](models/correct_split_all_augmentations/correct_split_all_augementations_lossplot.png)
![Loss Plot 2](models/correct_split_only_rotation/correct_split_only_rotation_lossplot.png)


## 3. Evaluation
```
pipeline/eval/eval.py
```
Hier haben wir mit den TestDaten evaluiert, also auf den 6 ungesehenen Bildern.

- Eval mit IoU auch so wie in den Notebooks (jaccard_score):
    - only rotation: (rotation zw. -3 und 3)
        - IoU: 0.7350793416367187
    - all augmentations: (rotation zw -3 und 3, je mit h, v, h+v)
        - IoU: 0.7995645707802462
    - reference: (das netz was die uns geben)
        - IoU: 0.1790 (was? :D)

Die Ergebnisse der Eval findet ihr unter: `pipeline/eval/results`:
![all_aug_eval](pipeline/eval/results/correct_split_all_augmentations/example_0.png)
![only_rot_eval](pipeline/eval/results/correct_split_only_rotation/example_0.png)
![reference_eval](pipeline/eval/results/reference/example_0.png)



Später ist mir aufgefallen, dass obwohl die IoU besser ist bei all
augmentations, dass das netz mit nur rotation aber besser tut beim wear
measurement!


## 4. Wear Measurement Preprocessing
```
pipeline/wear_measurement/preprocess.py
```
1. Aligning (hab ich mir nicht angeschaut, hab da straight benutzt was GPT mir
   gegeben hat und hat gut gefunzt, params könnt ihr aus der File rauslesen)
2. Cropping
3. Resizing
Ich habe die Bilder alle hier hochgeladen. Ihr könnt vllt ein gif machen aus
den unalignten bildern und den alignten, damit man den Unterschied sieht.
Liegen unter `data/test/Images/images` bzw. `data/test/Images/aligned`.


## 5. Wear Measurement Inferenz
```
pipeline/wear_measurement/inference.py
```
Hier findet die Inferenz statt. Anschließend werden die Predictions mit
find_contours auf den größte zusammenhängen Fetzen reduziert. Zuletzt findet
noch eine morphologische CLOSE und anschliessend OPEN operation statt.
Die ist sind sinnvoll, weil die labels teilweise löchrig sind. D.h. auch
ein zusammenhängender fetzen, kann noch löcher haben.
- Beim Close wird zuerst dilatiert und dann wieder mit dem selben kernel
  erodiert -> führt dazu, dass löcher geschlossen werden
- Beim Open genau andersrun also zuerst erodieren, dann dilatieren, führt dazu
  dass die inseln entfernt werden, eig unnötig, weil findcontours das schon
  macht aber habs tzd dringelassen, beide operationen auch dazu führen, dass
  die labels etwas glattkantiger werden, was auch eher zu der groundtruth
  passt.

Hier Beispielhaft eine Prediction einmal vor und nach dem Cleaning.
![Prediction](data/test/Images/predictions/pred_aligned_image0002500.png)
![Cleaned Prediction](data/test/Images/cleaned_predictions/pred_aligned_image0002500.png)

Ihr findet alle Predictions unter `data/test/Images/predictions` bzw.
`data/test/Images/cleaned_predictions`. Ich hab auch zusätzlich overlays
jeweils für alle predictions und alle cleaned predictions generiert, vllt wollt
ihr die auch benutzen, liegen in den entsprechenden ordnern.



## 6. Wear Measurement
```
pipeline/wear_measurement/wear_measurement.py
```
1. Hier zuerst den korrekten winkel finden mit dem wir messen wollen. Dazu
  finden wir die kanten mit canny edge detection in allen bildern
    - Wir nehmen den Median als gerade von der wir aus messen
    - In Wahrheit hab ich hier gepfuscht und den Winkel einfach auf 55.78Grad gesetzt
      weils nicht gescheit gefunzt hat und ich keine lust hatte mich damit
      auseinanderzusetzen :D (aber schreibt das mit dem winkel finden tzd gern in
      die präsi, hört sich besser an)
2. Danach suchen wir von der kante aus die breiteste stelle in der maske und
   das ist unser ergebnis je frame
3. die kanten sortieren (jeweils 4 aufeinanderfolgende bilder die zu den
   jeweiligen 4 schneiden gehören, und das im 8 frame takt, sieht man an den
   Bilder-Filenamen)
4. plotten der ergebnisse über die schnittnummern, einmal alles in einem diagramm
5. dann nochmal in einzelnen diagrammen, dort mit dbscan die outlier markiert,
   parameter für DBSCAN: `DBSCAN(eps=15, min_samples=2)`
Ergebnisse von diesen Schritten:
![Measurement example 1](data/test/Images/wear_measurement_old_pipeline/example_1.png)
![VBMax_plot_all](data/test/Images/wear_measurement/VBMax_plot_all.png)
![VBMax_plot_edge_1](data/test/Images/wear_measurement/VBMax_plot_edge_1.png)
![VBMax_plot_edge_2](data/test/Images/wear_measurement/VBMax_plot_edge_2.png)
![VBMax_plot_edge_3](data/test/Images/wear_measurement/VBMax_plot_edge_3.png)
![VBMax_plot_edge_4](data/test/Images/wear_measurement/VBMax_plot_edge_4.png)

## 7. Video
```
pipeline/video/preprocess.py
```
1. Preprocessing:
    - hier habe ich mich nicht an die vorgaben aus dem Notebook gehalten, die war unnötig.
    - ich bin stattdessen zur Extraktion folgendermaßen vorgegangen:
        - im 17. Frame sieht man das erste mal die 1. schneide gut.
        - danach folgt jedes 60. Frame wieder ein gut sichtbare schneide
        - die sind nicht perfekt aligned, aber genau das mache ich später ja
          mit dem image alignment, was wir vorher schon implementiert hatten
2. Danach halte ich mich genau an die pipeline von image processing, also
    1. `wear_measurement/preprocess.py` -> alignte bilder, cropped resized
    2. `wear_measurement/inference.py` -> predictions, cleaned_predictions
    3. `wear_measurement/wear_measurement.py` -> wear_measurement ergebnisse
Die Ergebnisse vom wear_measurement findet ihr unter
`data/test/Video/video0000030/wear_measurement` &
`data/test/Video/video0000031/wear_measurement`.
