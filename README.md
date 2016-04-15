# Association-Rules-Implementation-using-UCI-Datasets
Implementation of Association Rules using UCI Datasets
<br>

<br>
Implement the Apriori algorithm by first determining frequent itemsets and then proceeding to identify association rules. Consider that the input to your program is a
sparse matrix where the rows are transactions and columns are items. Each value in your matrix is a binary variable from {0, 1} that indicates presence of an item in the transaction.
<br>
a) Implement both Fk−1 × F1 and Fk−1 × Fk−1 methods. Allow in your code to track the
number of generated candidate itemsets as well as the total number of frequent itemsets.
<br>
b) Use three data sets from the UCI Machine learning repository to test your algorithms.
The data sets should contain at least 1000 examples, and at least one data set should contain 10,000
examples or more. You can convert any classification or regression data set into a set of transactions
and you are allowed to discretize all numerical features into two or more categorical features. Compare
these two candidate generation methods on each of the three data sets for three different meaningful
levels of the minimum support threshold (the thresholds should allow you to properly compare different
methods and make useful conclusions). Provide the numbers of candidate itemsets considered in a table
and discuss the observed savings that one of these methods achieves.
<br>
c) Enumerate the number of frequent closed itemsets as well as maximal frequent itemsets
on each of your data sets for each of the minimum support thresholds from the previous question.
Compare those numbers with the numbers of frequent itemsets.
<br>
d) Implement confidence-based pruning to enumerate all association rules for a given set of
frequent itemsets. Use the previous data sets, with three levels of support and three levels of confidence
to quantify the savings in the number of generated confident rules compared to the brute-force method.
<br>
e) For each data set and each minimum support threshold, select three confidence levels for
which you will generate association rules. Identify top 10 association rules for each combination of
support and confidence thresholds and discuss them (i.e. comment on their quality or peculiarity).
Select data sets where you can more easily provide meaningful comments regarding the validity of
rules.
<br>
f) Instead of confidence, use lift as your measure of rule interestingness. Identify top 10 rules
for each of the previous situations and discuss the relationship between confidence and lift.
