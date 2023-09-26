import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from svm_visualization import draw_boundary
from players import aaron_judge, jose_altuve, david_ortiz

fig, ax = plt.subplots()

aaron_judge['type'] = aaron_judge['type'].map({'S':1, 'B':0})

aaron_judge = aaron_judge.dropna(subset = ['plate_x', 'plate_z', 'type'])

plt.scatter(x = aaron_judge['plate_x'], y = aaron_judge['plate_z'], c = aaron_judge['type'], cmap = plt.cm.coolwarm, alpha = 0.5)
plt.xlabel('plate_x')
plt.ylabel('plate_z')

training_set, validation_set = train_test_split(aaron_judge, random_state = 1)

classifier = SVC(kernel = 'rbf')

two_columns = training_set[['plate_x', 'plate_z']]
classifier.fit(two_columns, training_set['type'])

draw_boundary(ax, classifier)

score = classifier.score(validation_set[['plate_x', 'plate_z']], validation_set[['type']])
print(score)





plt.show()





