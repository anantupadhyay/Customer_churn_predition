# CustomerChurnAnalysis

Project to analyze and predict which customers will leave or stay with company. Uses Deep Learning with Artificial Neural Networks to predict customer leave/stay probability.

Achieved an accuracy of order 0.871

# Files :
	ann.py -> Implementation of ANN using keras library
	alternate_churn_model.y -> Implementation of ANN with testing with one unseen tuple
	neural_net.py -> The classifier used for building app is built in this file using MLPC classifier
	app.py -> Contains code for building app(used flask for implementation).

	templates\
		home.html -> File containing the home page of the app
		result.html -> File for displaying the value returned by app.