Having seen the different parts of the Baby Name Classifier separately, you might be wondering how everything comes together to create a classifier that you can use for prediction. We will use the hashing, NaiveBayesPY, and NaiveBayesPXY functions and walk through example names. You will then implement the classifier in the final assignment.

Data
Let's start with checking what our data looks like. You will be given two files with train data girls.train and boys.train containing a baby name in each line. The assignment will also load two files with test data in the same format. For example, the girls.train file might look as follows (actual names can vary in the assignment):

Addisyn
Danika
Emilee
Aurora
Julianna
Sophia
Kaylyn
Litzy
Hadassah

Training
In the assignment, we will use +1 label for boys and -1 label for girls. This is the order in which we will create the classifier's constituent functions:

Hash each name to a -dimensional feature vector.
Compute NaiveBayesPY values  and .
Compute NaiveBayesPXY values  and  for all features .
Now we have trained our classifier. For prediction, we will compute the log likelihoods  and . This will then help us compute the predictions  and  using the Bayes' theorem.

Hashing
For this example, we will use 3 features:

 is 1 if first letter is 'a'
 is 1 if first letter is a vowel (one of 'aeiouy')
 is 1 if last letter is a vowel (one of 'aeiouy')
The feature vectors for these 4 names will be:




Alicia

1

1

1

Eve

0

1

1

Aiden

1

1

0

Fred

0

0

0

NaiveBayesPY
For simplicity, let's assume we are given 600 girls and boys names each. Therefore, .

NaiveBayesPXY
Using the train data, we can calculate what the probability is of a name starting with 'a', given that it is a boy's name. This is exactly .  is just  as probabilities of different values of the same feature given a label sums to 1. Empirically, we just find the fraction of boys' names that start with 'a'.

Similarly we can calculate the conditional probabilities of each feature given the name is a girl's.

We have calculated these empirical conditional probabilities from the train data for you:




 (boy)

0.09

0.17

0.30

 (girl)

0.16

0.24

0.73

Testing
To predict  — whether the following names are girls' or boys' — we find the log probabilities of each name being a girl's or a boy's and compare. If a name has a larger log probability for a girl (thus a larger probability for a girl), the classifier will predict -1 or "girl"; vice versa for boy.

For any label , .  Therefore, to predict a label  for a test point , we need to find  for different labels  and decide which log likelihood is the largest.

We calculated  in the NaiveBayesPY section, and  in the NaiveBayesPXY section.

If a test name's feature  is 1, then . If it is 0, then . Informally,  is the probability of the first feature value of the test name being whatever value it is, assuming it is a boy's name. We know probabilities for all possible values from the training data (computed in NaiveBayesPXY), and we just select the correct value.

 

Implementation tip: As all our features are binary, there is a clever way for computing :

 

where  is the value of . The above equation simplifies to our desired values subject to the values  takes. If , then , and if , then !

 





Alicia

0.09

0.17

0.30

-5.384

Eve

0.91

0.17

0.30

-3.070

Aiden

0.09

0.17

0.70

-4.537

Fred

0.91

0.83

0.70

-0.637





Alicia

0.16

0.24

0.73

-3.574

Eve

0.84

0.24

0.73

-1.916

Aiden

0.16

0.24

0.27

-4.569

Fred

0.84

0.76

0.27

-1.758

 

We now have all of the pieces. To make the final prediction, we add the log-posterior to each and compute the difference of logs. 

 


(boy)


(girl)

Prediction

Alicia

-6.077

-4.267

-1 (girl)

Eve

-3.763

-2.609

-1 (girl)

Aiden

-5.230

-5.262

1 (boy)

Fred

-1.330

-2.451

1 (boy)

 