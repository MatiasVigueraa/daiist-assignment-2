# Assignment 2 Report

*Delete this italic guidance as you fill in each section. You'll be asked to
defend any of this without your code in front of you. Write only what you
can actually explain.*

- **Name**: Matias Viguera Barrientos
- **Student ID**: 18542
- **Email**: mviguera.ieu2022@student.ie.edu
- **Group**: BBADBA 5A

## Business problem, and the cost of being wrong

*The decision this model supports, who or what acts on it, and what happens
when it's wrong in each direction. Be concrete about the asymmetry: which
mistake is worse, roughly by how much, and why. Everything downstream (your
threshold, how much you send to a person, which metric you care about) has to trace
back to this section.*

The business idea comes from gyms not being able to open 24/7 because they require someone on-site to constantly monitor the gym in case of issues. Usually, when lifting heavy weights, there might be an issue called syncope or lifting-induced syncope, that causes the lifter to black out. Moreover, there might be situations in which given the high-intensity workout someone might induce a cardiac arrest or due to natural conditions, something might happen that makes them fall and injure themselves. In these situations there should be someone present in the gym to assist, however, that means the gym has to pay additionally for an employee to keep watch on the gym during the night and changing shifts. So this is how I came up with this AI, with the idea of helping gyms that want to be open 24/7 with no staff on site at night, monitored remotely. This idea came from the Basic-Fit control room "unwell detection", but in this case we have a narrower prototype that detects fall postures. The only one who acts is the remote operator. With this the only decision that the remote operator needs to do is wether to keep looking if its something delicate or respond to the emergency. Therefore, the model doesn't call emergency services right away, it only flags the issue. 
Knowing this, this model considers missed falls as a member staying on the floor with nobody helping or coming. Usually minor falls, occasionally a serious collapse, plus injury claims, liability and reputation. And what a false alert means to us is the operator spending 2 minutes re-checking the camera and logging it. I have decided to set the missed fall at €25,000 because most falls in a gym tend to be minor incidents and the person gets up unaided, but a small share are serious collapses where if nobody comes there could actually be an issue. €25,000 sits at what in the spanish gym-injury could be considered upper end,  covering a typical injury while allowing for the occasional serious case, without pricing every fall as the worst outcome.

## Dataset

*Workspace / project / version on Roboflow Universe, and the link. What one
image represents, what the classes are, and why this dataset actually serves
the problem above rather than merely resembling it.*

The dataset is Patient Fall Detection, workspace medical-object-detection-zblyv, project patient-fall-detection, version 11 on Roboflow Universe (https://universe.roboflow.com/medical-object-detection-zblyv/patient-fall-detection/dataset/11), used as published. Each image is one frame from a fixed indoor camera, with one box per person labelled by posture: fallen, falling, sitting or standing. For the business decision, I have decided that fallen and falling count as a fall, sitting and standing are normal. 
This dataset really helped because its labels match the decisions we are looking for, operator needs to be able to decide wether in on the floor or going down, which are the classes. I chose this dataset over other options because it had the largest test split with balanced fall classes (53 fallen, 53 falling), while other options had only 2 falling examples in the test and SIREN had no test split. 
The biggest issue found in this dataset is that it only has staged falls which were done by actors in offices and living rooms, with no gym scenes and no floor exercises. So it just looks like a gym camera feed instead of reproducing an actual one.

## What the dataset's composition implies

*Instances per class, train/val/test sizes, typical object size relative to the
image. Then the interpretation: which per-class numbers will be too thin to
trust, whether the test split resembles the images you'd really see, and what
you therefore can't conclude from your results.*

| Split | Images | Images with a fall | fallen | falling | sitting | standing | Median box size |
|---|---|---|---|---|---|---|---|
| Train | 1,344 | 574 | 302 | 273 | 127 | 641 | 16.0% of image |
|Valid | 304 | 138 | 71 | 68 | 24 | 141 | 16.4% of image |
| Test | 244 | 106 | 53 | 53 | 25 | 112 | 15.3% of image |

In the images I used, the people used approximately 15-16% of the image, making it reasonably large and clearly visible. The test numbers are  show to be thin in the parts, sitting with only 25 examples, making it that each mistake moves the score by 4 points, then each of the 53 fallen images counts for near 2 points of recall. What is most important is that for costs, there are only 138 images that are considered normal test images meaning that a false-alert rate below roughly 1 in 138 cannot be measured. 
Then the test split looks like the training data, 136 of 244 test images match the training image's fingerprint, then inspecting them shows frames that connect as a video, in same room and cameras with actors. The falls appear in 43% of test images, against roughly 0.0006% in the real scenario, which i why all costs are re-weighted to an assumed base rate. Some labels look inconsistent, there is one validation image that was skipped for mixing polygon and boxed and then 8 trainings with 2 test polygon labers were converted to boxes.

So with this information I cannot get enough information to say that the model would perform correctly in new cameras in a real gym, neither if it would raise false alerts or how it would handle floor exercises such as planks and streches, none of which appear in the data. The test results are a description of familiar rooms and should be read as what it is.

## Finetuning and baselines

*Which model and task (detection or classification) and why. A results table on
the same test split: your finetuned model, the pretrained COCO model zero-shot,
and a trivial baseline. What changed, and whether the gap is as large as you
expected.*


The model is a detection type model. The objective is to detect when someone falls during the gym hours were no employee from the gym is present in person. Moreover, each person needs a posture and an image may contain multiple people in it, so each one needs its own box with detected posture. In this case I fine-tuned the model yolo11n, the smallest model from the YOLO11 family models, using the pre-trained COCO weights. This one counts with 1300 images for training and it trained for about an hour in a laptop.

| Setting | Value |
|---|---|
| Model | yolo11n (pretrained on COCO), object detection |
| Epochs / image size / batch / seed | 50 / 640 / 16 / 0 |
| Where trained | MacBook Pro M1 Pro, Apple GPU (MPS), about 1 h 7 min |
| Weights | weights/best.pt (best epoch on validation) |

In this case, the COCO model zero-shot has no "fall" class, therefore the rule used is when the box of the person is wider than it is tall", that would be considered someone lying down, therefore counts as "fall". The trivial baseline predicts nothing. With all the three compared in the same 244 images, with a 0.5 threshold:

| Model (same 244 test images, threshold 0.5) | Falls caught | False alerts | Falls missed | Recall | Precision | Average precision |
|---|---|---|---|---|---|---|
| Fine-tuned yolo11n | 103 | 3 | 3 | 0.972 | 0.972 | 0.990 |
| COCO zero-shot ("person wider than tall") | 12 | 5 | 94 | 0.113 | 0.706 | 0.652 |
| Trivial (predict nothing) | 0 | 0 | 106 | 0 | 0 | n/a |

At the detection level, the fine-tuned model shows to get mAP50 0.972 and mAP50-95 0.663 on the test set. Regarding the gap against COCO, it is way above what I expected, the rule I set of "wider than tall" is just catching 12 of 106 falls. This could be happening because when mid-fall there might be still some upright position that is interefing with the results, another cause might be that overhead cameras can have issues when recording the box wider than taller. In this case, the fine-tuning is able to teach the model what an actual person falling or that fell looks like instead of just relying on the box. Although, the test score is higher than the validation score (0.972 vs 0.952), this does make sense since it fits with the idea of overlap between the images from the test and train. Therefore, we can say that this results are really good.

## Per-class evaluation and failure modes

*Per-class AP50, or per-class precision/recall and the confusion matrix.
Which class carries the headline number, which one is dragging, and what the
model actually confuses with what. Show where it breaks, not just that it
works.*

[Table1]
We can see that the mAP50 of 0.972 is mainly guided by standing and sitting, which its fine for us and also standing is the largest class in the dataset, so it makes sense. In this case, we are getting that the class that is actually important for us, fallen, is the one dragging it down, showing a low precision (0.856) and a mAP50-95 (0.604). 

[Table2]
This confusion matrix shows that the model is not mistaking a fall for standing or sitting, the main errors are coming from the mix of fallen and falling, which in the end is not affecting the decision since it is set for both to flag and trigger the alert to the monitoring responsible. However, there are false positives, were 15 fall were detected and a box was drawn when no one was labelled. Finally there was only one person standing that was missed entirely.

Another issue or something to take into account is regarding the confidence the model shows. For instance, in one of the images there is a person clearly falling down, the model marks it as falling with a confidence score of 0.78, which is below to a later threshold of 0.80 I defined later. In a gym, the possible failures could come more from people doing planks, stretches and situps, exercises that force people to get in the floor, given that the model has never seen them, it could flag them as falls. 

## Cost matrix and operating point

*Your euro cost for each kind of error and where each figure came from. The
base rate you assumed. The cost-against-threshold sweep, and the operating
point you chose. State the assumption that would most change your answer if it
were wrong.*

[Table1]



## What a person still has to check

*Your two thresholds, and what share of cases they send to a person. How many
that is per day, how long one takes to check, and how many people that needs.
Is that realistic? Then your design priced against the three alternatives:
doing nothing, having a person check everything, and letting the model decide
on its own.*

## Limitations & next steps

*Real limitations you found, and concretely how you'd address each one with
more time or data. Not generic hedging.*

## Generative AI use disclosure

*Per the syllabus AI Policy: what you used and how, or "no AI content used."*
