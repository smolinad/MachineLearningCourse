### A Pluto.jl notebook ###
# v0.19.23

#> [frontmatter]
#> title = "Code Assignment #1 - Binary Classification"
#> date = "2023-03-07"

using Markdown
using InteractiveUtils

# ╔═╡ 02ef7e5c-0858-4da4-af5e-1dcc6060c6ef
begin
	
	using CSV
	using DataFrames
	using Dates
	
	using LinearAlgebra
	
	using StatsPlots
	import PlotlyJS as pltjs

	using GLM
	using MLJ
	using MLJLinearModels
	using DecisionTree
	
end;

# ╔═╡ dff3c71e-2ed4-4663-8441-e10b1b2f76d6
md"# Code Assignment \# 1 - Binary Classification 
*Universidad Nacional de Colombia*\
Sebastian Molina  \
[smolinad@unal.edu.co](mailto:smolinad@unal.edu.co)"

# ╔═╡ 8fb0a34b-80b0-4862-ae94-0a777e430ff2
md"
- Make all the models compete.
- Partition the datasets into train, test, eval."

# ╔═╡ 9786047f-c383-4269-af6f-90039f3caf74
md"## Problem Identification

In this notebook we will be solving binary classification problems with labeled input. Therefore, we are solving this classification problem as supervised learning, specifically using Support Vector Machines (SVM), Logistic Regression, and Decision Trees."

# ╔═╡ 958e8517-f147-4a61-a3c1-3335f8e8c36d
md"## Importing the packages
In this notebook we will use `MLJ`, `MLJLinearModels` and `DecisionTree` packages for data processing, Logistic Regression and Decision Tree models, respectively."

# ╔═╡ fbd49b80-c4e9-4b8d-83d7-ac3c3c958a38
md"## Utility functions
Here we define some utility functions that will be used recurrently in this notebook."

# ╔═╡ db362343-b3be-4a2d-b4b0-002417560b3a
md"### `dataSplit` function
This function splits a `DataFrame` into training an testing into a 80/20 split. Then, it will convert this dataframes into matrices, which is the common input for this models."

# ╔═╡ 48a07b04-52e6-4dbe-beb5-5b45e148136e
function dataSplit(db, toMatrix=true)
	train, test = MLJ.partition(
		db, 0.8, 
		rng=123, 
		shuffle=true)
	
	if toMatrix
		Matrix(train), Matrix(test)
	else
		train, test
	end
end;

# ╔═╡ 604677b8-ec66-4137-ab47-0121874457bb
function dataSplit2(db, toMatrix=true, shuffle=true)
	train, val, test = MLJ.partition(
		db, 
		0.7, 
		0.2, 
		rng=123, 
		shuffle=shuffle)
	
	if toMatrix
		Matrix(train), Matrix(val), Matrix(test)
	else
		train, val, test
	end
end;

# ╔═╡ 00540701-8536-4603-a337-05ca3df89ad7
md"## SVM Implementation with Regularization"

# ╔═╡ 5837be60-44d6-4219-b0b3-940de66337de
md"### `SVMRegular` struct (class)
- `svm.w`: It is the weights vector, which in part defines the decision bound.
- `svm.b`: Known as *bias*, it is the displacement of the decision bound.
- `svm.lr`: This parameter is the learning rate of the gradient descent algorithm, which determines the step size towards the minimum of the cost function in each iteration. 
- `lambda`: This hyperparameter is related to the regularization of the weights vector.
- `svm.num_iters`: Number of iterations to make and find the decision bound.
- `svm.num_features`: It is the number of features (coordinates or dimensions) in a feature set (data vector). From the provided databases, it will be the number of columns minus one."

# ╔═╡ b5645969-e265-46d5-b676-1404b1b9d69c
mutable struct SVMRegular
	w::Array
	b::Float64

	lr::Float64
	lambda::Float64
	num_iters::Int
	
	num_features::Int
	
	function SVMRegular()
		new([], 0.0, 0.001, 0.01, 5000, 0)
	end
end

# ╔═╡ 6519e8ad-ea5c-428a-90be-a81ef87195c0
md"### `fit` method"

# ╔═╡ 9ee89705-ad5a-465a-b704-2dab939c3ed7
md"
The goal of the Support Vector Machine (SVM) algorithm is to find a decision boundary hyperplane $\mathbf{w} \cdot \mathbf{x} - b = 0$ which separates the data into two classes, while maximizing the distance between the two possible clusters of input data. For a given input data vector $\mathbf{x}_i$ and its corresponding label $y_i$, this problem is designed by the following constraints:

$\begin{cases}
\mathbf{w} \cdot \mathbf{x}_i - b \leq -1,\ y_i = -1\\
\mathbf{w} \cdot \mathbf{x}_i - b \geq +1,\ y_i = +1\\
\end{cases}.$

The preceding constraints can be resumed into 

$y_i(\mathbf{w} \cdot \mathbf{x}_i) \geq +1.$

By the [distance between parallel hyperplanes formula](https://metric.ma.ic.ac.uk/metric_public/vectors/vector_coordinate_geometry/distance_between_planes.html), we know that the distance between the suporting hyperplanes which intercept the closest data points in different clusters is given by $D = \frac{2}{|w|}$. As we want to maximize $D$, we should minimize $|w|$.

#### Hinge function
We need a systematic way to find the $\mathbf{w}$ vector, so we introduce the following loss function. In case the label is predicte incorrectly,

$L_{\mathbf{w}, b}(\mathbf{x}_i, y_i) = \begin{cases}
0&,\ y_i(\mathbf{w} \cdot \mathbf{x}_i - b) \geq +1\\
1 - y_i(\mathbf{w} \cdot \mathbf{x}_i - b)&,\ \text{otherwise} 
\end{cases}.$

#### Regularization
This is a method that yields a simpler and maybe inexact result, opposed to a complex and more exact result. For this prblem, consider the following cost function:

$J = \lambda|\mathbf{w}|^2 + \frac{1}{N}\sum_{i=1}^{N} L_{\mathbf{w}, b}(\mathbf{x}_i, y_i).$

Observe that for the $i$-th vector data $\mathbf{x}_i$ and its corresponding label $y_i$, we have that

$J_i = \begin{cases}
\lambda|\mathbf{w}|^2&,\ y_i(\mathbf{w} \cdot \mathbf{x}_i - b) \geq 1\\
\lambda|\mathbf{w}|^2 + 1 - y_i(\mathbf{w} \cdot \mathbf{x}_i - b)&,\ \text{otherwise}
\end{cases}.$

As we want to minimize cost, using [gradient descent](https://en.wikipedia.org/wiki/Gradient_descent) we can create update rules for $\mathbf{w}$ and $b$ for every iteration of the minimization process. Therefore,

$\begin{cases}
\frac{\partial J_i}{\partial \mathbf{w}} = 2\lambda\mathbf{w} \!\!\!\!\!&,\ \frac{\partial J_i}{\partial b} = 0&&,\ y_i(\mathbf{w} \cdot \mathbf{x}_i - b) \geq 1\\
\frac{\partial J_i}{\partial \mathbf{w}} = 2\lambda\mathbf{w} - y_i\mathbf{x}_i \!\!\!\!\!&,\ \frac{\partial J_i}{\partial b} = y_i&&,\ \text{otherwise}
\end{cases}.$

Using the previous results, we can create the update rules including the learning rate parameter $\gamma$, as follows:

$\begin{cases}
\mathbf{w} = \mathbf{w} - \gamma(2 \lambda \mathbf{w}) \!\!\!\!\!&,\ b = b &&,\ y_i(\mathbf{w} \cdot \mathbf{x}_i - b) \geq 1\\
\mathbf{w} = \mathbf{w} - \gamma(2 \lambda \mathbf{w} - y_i\mathbf{x}_i) \!\!\!\!\!&,\ b = b - \gamma y_i &&,\ \text{otherwise}
\end{cases}.$

Let's implement this algorithm into the `fitSVM` function.
"

# ╔═╡ a587a40c-582b-48fc-a3bf-aed21c9c5875
function fitSVM(svm::SVMRegular, data::Matrix)
	
	svm.num_features = size(data)[2] - 1

	svm.w = repeat([0], svm.num_features)
	svm.b = 0.0
	
	for _ in range(1, svm.num_iters)
		for row in eachrow(data)
			if (last(row) * (dot(svm.w, row[1:svm.num_features]) - svm.b)) >= 1
				svm.w -= svm.lr * (2 * svm.lambda .* svm.w)
			else
				svm.w -= svm.lr * (2 * svm.lambda .* svm.w - (row[1:svm.num_features] .* last(row)))
				svm.b -= svm.lr * last(row)
			end
		end
	end
		
end;

# ╔═╡ 72484660-f9ea-4be1-bc76-f15102cc2515
md"### `predictSVM` function
For the `predict` method we just only evaluate the loss function to predict the label and count the number of misses with respect to the real label. From here on, we will use 🔴 for $-1$ and 🔵 for $1$."

# ╔═╡ d5537edd-28ec-46b0-b9d9-f347b772ccb2
function predictSVM(svm::SVMRegular, features)
	
	[(sign(dot(row, svm.w) + svm.b) < 0 ? "🔴" : "🔵") for row in eachrow(features)]
	
end;

# ╔═╡ b4206d72-52ab-44ec-8336-027762356df7
function resultsSVM(svm::SVMRegular, data)
	
	result = Matrix{Float64}(undef, 0, 2)
	
	for row in eachrow(data)
		prediction = sign(dot(row[1: svm.num_features], svm.w) - svm.b)
		result = vcat(result, [prediction last(row)])
	end

	misses_red = 0
	misses_blue = 0
	total = 0
	
	for res in eachrow(result)
		total += 1
		if res[1] != res[2]
			if res[2] == -1
				misses_blue += 1
			elseif res[2] == 1
				misses_red += 1
			end
		end
	end

	misses_blue_pc = round(misses_blue/total, digits=3)
	misses_red_pc = round(misses_red/total, digits=3)
	accuracy = round(1 - misses_blue_pc - misses_red_pc, digits=3)
	
md"
- 🔴 (-1) Misses: $(misses_blue_pc*100)%
- 🔵 (1) Misses: $(misses_red_pc*100)%
- Accuracy: $(accuracy*100)%
- Number of rows: $(total)
"
	
end;

# ╔═╡ b9e43bd4-6d88-4b08-8a3a-fabef87a0ece
md"### `plotSVM` function
The `plotSVM` function only works data with a maximum of 3 features, and it plots the decision boundary and supporting vectors of the SVM. The code cell can
be hidden for readability of the notebook."

# ╔═╡ 53fb9803-ae4d-47ad-b1cd-d354d119afda
function plotSVM(svm, data, new_data)
	
	if svm.num_features == 2
		plotlyjs()

		min_X = minimum(vec(
			vcat(Matrix(data)[:,1:1], Matrix(new_data)[:,1:1])
		)) - 10
		
		max_X = maximum(vec(
			vcat(Matrix(data)[:,1:1], Matrix(new_data)[:,1:1])
		)) + 10
		
		q = plot()
		
		bound2d(x) = -(svm.w[1]*x + svm.b) / svm.w[2]
		red_sv2d(x) = -(svm.w[1]*x + svm.b + 1) / svm.w[2]
		blue_sv2d(x) = -(svm.w[1]*x + svm.b - 1) / svm.w[2]
		
		plot!(q, bound2d, min_X, max_X,
			label="Decision bound", lc=:black, lw=2)
		
		plot!(q, red_sv2d, min_X, max_X,
			label="Support vector -1", lc=:red, lw=2)
		
		plot!(q, blue_sv2d, min_X, max_X, 
		 	label="Support vector +1", lc=:blue, lw=2)
		
		@df data scatter!(q, :x1, :x2, color=:x3, label=false)
		
		@df new_data scatter!(q, :x1, :x2, color=:gold, label="New data")
		
	elseif svm.num_features == 3
		
		plotlyjs()

		p = plot()
		
		bound3d(x, y) = -(dot(svm.w[1:2], [x y]) + svm.b) / svm.w[3]
		red_sv3d(x, y) = -(dot(svm.w[1:2], [x y]) + svm.b + 1) / svm.w[3]
		blue_sv3d(x, y) = -(dot(svm.w[1:2], [x y]) + svm.b - 1) / svm.w[3]
		
		surface!(p, -100:1:100, -100:1:100, bound3d, 
			color=:green, opacity=0.7, label="Decision bound")
		
		surface!(p, -100:1:100, -100:1:100, red_sv3d, 
			color=:red, opacity=0.5, label="SV -1")
		
		surface!(p, -100:1:100, -100:1:100, blue_sv3d, 
			color=:blue, opacity=0.5, label="SV +1")
		
		@df data scatter!(p, :x1, :x2, :x3,
			color=:x4, 
			label=false,
			markersize=2)
		
		@df new_data scatter!(p, :x1, :x2, :x3, 
			color=:gold, 
			label="New data",
			markersize=2)

	else
		
		md"Dimensions of representation space are too big to be plotted."
		
	end	
	
end;

# ╔═╡ e761d48f-6b04-4c8d-be81-18edc017e95e
md"## Examples
Here we will use the example provided in the Jupyter Notebook, with a little modification for 3D."

# ╔═╡ e1bcc629-1c9c-4d8e-90f5-ada601350fef
md"### Example 1: 2D version"

# ╔═╡ 4600aa8f-b941-4939-97a2-ff8da9d27e8d
example_1 = [1 7 -1
		   2 8 -1
		   3 8 -1
		   5 0 1
		   6 -1 1
		   7 3 1];

# ╔═╡ 904d7e4d-9ca1-4d36-b7fd-cfe0793d77e1
begin
	svm_example_1 = SVMRegular()
	fitSVM(svm_example_1, example_1)
end

# ╔═╡ c3005d9d-06c7-4cf1-8f02-7bed71f9b03c
begin
	new_example_1 = [
		0 8;
		5 -5
	]
	
	predictSVM(svm_example_1, new_example_1)
end

# ╔═╡ e0ae4760-94eb-45ae-90d1-a6ece8d49df3
resultsSVM(svm_example_1, new_example_1)

# ╔═╡ 0c6bf5ba-00cd-47bc-bae8-4b93b2732cc5
md"Graphically, we can see that in fact the SVM is making a good job at finding the decision boundary."

# ╔═╡ fd3b7a31-5655-4a98-b1f5-e4a5deca2f4e
md">##### ⚡ Note that the following plot is interactive! You can drag the camera, zoom in, and select the points using your mouse/mousepad. Please, try it!"

# ╔═╡ ce6e9347-5974-4bb4-aba8-bfea8b17d751
plotSVM(svm_example_1, DataFrame(example_1, :auto), DataFrame(new_example_1, :auto))

# ╔═╡ beae22b7-3fbb-429f-85e3-d7fd4ac82d5f
md"### Example 2: 3D version"

# ╔═╡ f19de1fb-6512-479f-88a6-7e9cf84a3849
example_2 = [
	1 7 -4 -1
	2 8 -5 -1
	3 8 -4 -1
	5 1  2 1
	6 -1 3 1
	7 3  5 1];

# ╔═╡ 1d56f85e-2df2-4ee0-8e91-4b435f8baa76
begin
	svm_example_2 = SVMRegular()
	fitSVM(svm_example_2, example_2)
end

# ╔═╡ 9e27cd53-d648-4a06-b47e-39ec06a1ae02
begin
	new_example_2 = [
		0 8 -4;
		5 -5 3
	]
	
	predictSVM(svm_example_2, new_example_2)
end

# ╔═╡ eeb04698-1cb7-4c26-a1db-966fd57142f4
resultsSVM(svm_example_2, new_example_2)

# ╔═╡ 2c9f360e-fc82-4f10-b192-a1785465add5
md">##### ⚡ Note that the following plot is interactive! You can drag the camera, zoom in, and select the points using your mouse/mousepad. Please, try it!"

# ╔═╡ 124c13a2-5682-49c3-a7e3-b20f405d27a7
plotSVM(svm_example_2, 
	DataFrame(example_2, :auto), 
	DataFrame(new_example_2, :auto)
)

# ╔═╡ b3d9f580-5b85-40f7-8864-0cda14b4a8d4
md"## Database 1 - Banknote Authentication Data Set"

# ╔═╡ eb46ddaf-4cb2-4cb6-a187-49e3d27e48f0
md"### Feature Description
As explained in [Discrete Wavelet Transform](https://www.sciencedirect.com/topics/mathematics/discrete-wavelet-transform), 
>(...)the discrete wavelet transform (DWT) is a transform that decomposes a given signal into a number of sets, where each set is a time series of coefficients describing the time evolution of the signal in the corresponding frequency band.

The features present in the given dataset are the second, third and fourth moments of the Wavelet transform over the images, the (Shannon) entropy of the image, and the label, respectively:
- *Variance of Wavelet Transformed image (continuous)*
- *Skewness of Wavelet Transformed image (continuous)*
- *Kurtosis of Wavelet Transformed image (continuous)*
- *Entropy of image (continuous)*
- *Class (Label)*
"

# ╔═╡ afbac097-eac1-4b9b-a9d3-30783f84be24
md"### Data processing"

# ╔═╡ 333c9e3b-6f1d-4993-9368-190c8527a86d
md"Here we download the first database, obtained from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/banknote+authentication)."

# ╔═╡ c4165e26-7fd6-4833-9c09-e28b7be27b01
banknote_raw = download("https://archive.ics.uci.edu/ml/machine-learning-databases/00267/data_banknote_authentication.txt");

# ╔═╡ c590c041-dd90-4f3d-895f-ff0607a0d0ee
begin
	banknote_df = CSV.read(banknote_raw, 
		DataFrame, 
		header=["Variance", "Skewness", "Kurtosis", "Entropy", "Label"]
	)
	
	banknote_df
end

# ╔═╡ 72268cf2-9a1c-40e3-a5e7-e3ac7851dc70
md"#### Data split
Here we split the data into training, validation and testing, leaving the resulting split as `DataFrames`.
"

# ╔═╡ e95f78d2-9960-4c6b-837d-4c1af186bcb0
begin
	banknote_train, banknote_val, banknote_test = dataSplit2(banknote_df, false)
end;

# ╔═╡ d7820f71-2a92-4693-9dc4-fcfd5ecb05a5
md"### SVM"

# ╔═╡ 94bb684a-81d8-4255-887b-4f00edee3d0d
md"Here, we change the label $0$ to $-1$, as needed by the SVM algorithm. Observe that the label is in the fifth column, and in general, we will format data so that the label column is the $n$-th column. Next, we proceed to split the data into training and testing, with an 70/20/10 split. There are $(size(filter(row -> row.Label == 0, banknote_df))[1]) items labeled as 🔴, and $(size(filter(row -> row.Label == 1, banknote_df))[1]) items labeled as 🔵."

# ╔═╡ f78da47b-15db-474b-86b7-163378bc6e95
begin
	svm_banknote_train = deepcopy(banknote_train)
	svm_banknote_val = deepcopy(banknote_val)
	svm_banknote_test = deepcopy(banknote_test)

	replace!(svm_banknote_train[!, "Label"], 0 => -1)
	replace!(svm_banknote_val[!, "Label"], 0 => -1)
	replace!(svm_banknote_test[!, "Label"], 0 => -1)
end;

# ╔═╡ 4e6cf955-5d1b-4eff-8243-e3826650ca6b
begin
	svm_banknote = SVMRegular()
	fitSVM(svm_banknote, Matrix(svm_banknote_train))
end

# ╔═╡ de6c5c60-e2d5-4eef-87c0-facd98c447cd
resultsSVM(svm_banknote, Matrix(svm_banknote_val))

# ╔═╡ 8bd3dabc-3a37-477c-ae71-dbf34bd16c4b
md"### Logistic Regression"

# ╔═╡ cbf222b6-3cac-40b6-9d75-fe0ba7a0058a
begin
	fm_banknote = @formula(Label ~ Variance + Skewness + Kurtosis + Entropy + Label)
	lr_banknote = glm(fm_banknote, banknote_train, Binomial(), ProbitLink())
end;

# ╔═╡ 49a33464-8e04-4238-ba7c-d16dc1acf101
md"Here, we apply a threshold function to each value of the prediction, so the prediction corresponds to $0$ or $1$."

# ╔═╡ 6c1290ec-cccb-4a99-b2d9-9843cb1a56d2
lr_banknote_predict = 
	(x ->x < 0.5 ? 0. : 1.).(GLM.predict(lr_banknote, banknote_val));

# ╔═╡ 9c33577c-cb72-47c8-b3f9-11a9ee2d0536
md"Finally, we calculate the accuracy by comparing the prediction and the test data labels, and dividing the number of correct comparisons with the testing data size. "

# ╔═╡ 4984ee85-31fe-43f1-a606-d597cb9d58b2
lr_banknote_accuracy = 
	sum(lr_banknote_predict .== vec(banknote_val[:,end])) / size(banknote_val, 1);

# ╔═╡ 715d2091-a709-4559-abfd-0bbe86a70781
md"From the last result we can see that the accuracy of the logistic regression model for this dataset is $(lr_banknote_accuracy*100)%."

# ╔═╡ 2febf6b3-477f-44e1-a204-9f6a892fa1cd
md"### Decision tree"

# ╔═╡ 9458e108-925a-45f5-8669-5ad735859b4e
md"As with the previous models, we split the data, then we fit the model, and finally predict with the model to get an accuracy statistic."

# ╔═╡ b6f9b431-82ff-44d0-b374-7e6a8ad401e2
dt_banknote = DecisionTreeClassifier();

# ╔═╡ e3311988-cc8c-476d-8f6e-3a37ed1d48df
DecisionTree.fit!(
	dt_banknote, 
	Matrix(banknote_train)[:,1:end-1],
	Matrix(banknote_train)[:,end]
);

# ╔═╡ b479eb87-dcbe-4171-a03a-9153832d2139
dt_banknote_prediction = DecisionTree.predict(
	dt_banknote, Matrix(banknote_val)[:,1:end-1]
);

# ╔═╡ f1a4ba68-9a48-49a9-8760-3521cab29896
dt_banknote_accuracy = 
	sum(dt_banknote_prediction .== vec(Matrix(banknote_val)[:,end])) / size(banknote_val, 1);

# ╔═╡ c888d313-634b-4d5c-bc31-184f3efaa884
md"From the last result we can see that the accuracy of the decision tree model for this dataset is $(round(dt_banknote_accuracy, digits=3)*100)%."

# ╔═╡ 6df86af7-2804-4961-94da-cffed0410125
md"### Testing

From the previous results, the model with the best accuracy was the linear regression. Then we proceed to test the model."

# ╔═╡ e4f5b9d5-06b2-4296-9694-ef97ef0b2c5f
md"## Database 2 - Occupancy Detection Data Set"

# ╔═╡ 7886a080-ef8f-4bf0-9f5c-045e1636eece
md"Again, the following database was obtained from [UCI Machine Learning Repostory](https://archive.ics.uci.edu/ml/datasets/Occupancy+Detection+)."

# ╔═╡ 7e05295f-3e1e-4800-a989-ae81f7b59de0
md"### Feature Description
- *Date*: Formatted as `year-month-day hour:minute:second`.
- *Temperature, Celsius*: It the room temperature, at a given `date`.
- *Relative Humidity, %*:
- *Light, in Lux*: According to [Merriam-Webster](https://www.merriam-webster.com/dictionary/lux), a lux is the unity of measure of illumination over a surface, equialent to a lumen per square meter.
- *CO2, in ppm*: It is the quantity of CO2 present in the room, at a given `date`.
- *Humidity Ratio*: Derived quantity from temperature and relative humidity, expressed in terms of $\frac{\mathrm{kg}_{\mathrm{water\ vapor}}}{\mathrm{kg}_{\mathrm{air}}}$.
- *Occupancy*: 0 (changed to -1) for not occupied, 1 for occupied status (for a given room)."

# ╔═╡ b0da2855-91a4-463a-bbda-ab8d8160b850
md"### Data processing"

# ╔═╡ 57149050-83ad-42c4-807a-616deec1ae8c
begin
	occupancy_df = CSV.read("./datatraining.txt", DataFrame, header=true)
	
	occupancy_date_df = deepcopy(occupancy_df)
	
	select!(occupancy_df, Not([:Column1, :date]))
	select!(occupancy_date_df, Not([:Column1]))
	
	occupancy_df
end

# ╔═╡ 01eb7746-2ab2-45b3-a602-904f930e1b23
md"As before, we proceed to split the data into training and testing, with an 80/20 split. There are $(size(filter(row -> row.Occupancy == 0, occupancy_df))[1]) items labeled as 🔴, and $(size(filter(row -> row.Occupancy == 1, occupancy_df))[1]) items labeled as 🔵. Clearly, this is a moderately imbalanced dataset, as $(round(size(filter(row -> row.Occupancy == 0, occupancy_df))[1]/size(occupancy_df, 1), digits=3)*100)% of total data is from one class."

# ╔═╡ 74ac32f4-e44b-4dea-b239-dc2ee39c734f
begin
	occupancy_test_raw = CSV.read("./datatest.txt", DataFrame, header=true)
	select!(occupancy_test_raw, Not([:Column1, :date]))

	occupancy_test_raw
end

# ╔═╡ b305634d-e636-4fd1-b235-c9174b9c7b52
md"Checking the provided testing data set, there are $(size(filter(row -> row.Occupancy == 0, occupancy_test_raw))[1]) items labeled as 🔴, and $(size(filter(row -> row.Occupancy == 1, occupancy_test_raw))[1]) items labeled as 🔵. However, we decided to sample the training data set to get the testing data, as we don't know if the provided testing data is included in the training data."

# ╔═╡ 8004fe42-7406-4df9-bd27-1f9a7b4c8b07
md"### SVM"

# ╔═╡ f4551c02-98a1-4485-abc5-2991483ecdaa
begin
	svm_occupancy_df = deepcopy(occupancy_df)
	replace!(svm_occupancy_df[!, "Occupancy"], 0 => -1)
	svm_occupancy_train, svm_occupancy_test = dataSplit(svm_occupancy_df)
end;

# ╔═╡ 909791fe-b893-4597-aeb9-499019ebeca7
begin
	svm_occupancy = SVMRegular()
	fitSVM(svm_occupancy, svm_occupancy_train)
end

# ╔═╡ b375353a-0bd1-4bd0-a5f7-fe60ddad626b
resultsSVM(svm_occupancy, svm_occupancy_test)

# ╔═╡ 49873af1-4979-4962-a69a-9006af80bfa5
md"### Logistic Regression"

# ╔═╡ e9f4f135-588e-4679-96f2-0fb8a9bdf016
lr_occupancy_train, lr_occupancy_test = dataSplit(occupancy_df, false);

# ╔═╡ 79b23ab0-dbe1-4290-b095-36772bb09363
begin
	fm_occupancy = 
		@formula(Occupancy ~ Temperature + Humidity + Light + CO2 + HumidityRatio)
	lr_occupancy = glm(fm_occupancy, lr_occupancy_train, Binomial(), ProbitLink())
end;

# ╔═╡ 3a27f686-8612-473c-a005-3c452df03237
lr_occupancy_predict = 
	(x ->x < 0.5 ? 0. : 1.).(GLM.predict(lr_occupancy, lr_occupancy_test));

# ╔═╡ abdfcc54-1652-4169-9b0a-af2257d2ef8e
lr_occupancy_accuracy = 
	sum(lr_occupancy_predict .== vec(lr_occupancy_test[:,end])) / size(lr_occupancy_test, 1);

# ╔═╡ a1317f7f-a939-4421-b31f-775f1e943123
md"From the last result we can see that the accuracy of the logistic regression model for this dataset is $(round(lr_occupancy_accuracy, digits=3)*100)%."

# ╔═╡ 160b93db-1455-4319-99aa-db19780638aa
md"### Decision tree"

# ╔═╡ d1cbb5da-44db-40dd-b1b8-745d0ee77f82
dt_occupancy_train, dt_occupancy_test = dataSplit(occupancy_df);

# ╔═╡ 2cdb6e50-19ea-4fc1-9bc8-80a5fb30a528
dt_occupancy = DecisionTreeClassifier();

# ╔═╡ 3a847626-61a7-4d0e-87c1-e8676a6406f5
DecisionTree.fit!(
	dt_occupancy, 
	dt_occupancy_train[:,1:end-1],
	dt_occupancy_train[:,end]
);

# ╔═╡ 8c70a0bf-4346-4fd0-8d77-61008be3773b
dt_occupancy_prediction = DecisionTree.predict(
	dt_occupancy, dt_occupancy_test[:,1:end-1]
);

# ╔═╡ 1c53a451-b1dd-41b7-80cd-cb8da81737f1
dt_occupancy_accuracy = sum(
	dt_occupancy_prediction .== vec(dt_occupancy_test[:,end])
) / size(dt_occupancy_test, 1);

# ╔═╡ 806708b8-5e43-4591-9eeb-38db217c6a01
md"From the last result we can see that the accuracy of the logistic regression model for this dataset is $(round(dt_occupancy_accuracy, digits=3)*100)%."

# ╔═╡ a07b9a55-a745-4c54-9393-d4b8752475a2
md"# DELETE FROM HERE!!!!!!!"

# ╔═╡ 43a09104-5948-49ce-9a7b-7b938ffb26d3
md"## Extra
### Database 2 - Occupancy Detection Data Set (taking into account the date)"

# ╔═╡ 52eb82fb-97b6-42be-a9bb-63a1f8624c45
md"As a bonus, we will take into account the date and time on where the measures were taken. We proceed to split the date and time into the corresponding `Date`, `Hour`, `Minute`, and `Second` columns."

# ╔═╡ 91ee8380-7d05-47c1-a88d-80e9b09195e3
DataFrames.transform!(occupancy_date_df, 
	:date => ByRow(x -> DateTime(x, dateformat"y-m-d H:M:S")) => :date);

# ╔═╡ c3ba53fe-ccef-4dd1-88f4-72365c880603
begin
	
	DataFrames.transform!(occupancy_date_df, 
		:date => ByRow(
			x -> (
				convert(AbstractFloat, Dates.day(x)), 
				convert(AbstractFloat,Dates.hour(x)), 
				convert(AbstractFloat,Dates.minute(x)),
				convert(AbstractFloat,Dates.second(x))
			)
		) => [:Day, :Hour, :Minute, :Second])
	
	select!(occupancy_date_df, Not(:date))

	select!(occupancy_date_df, 
		[:Temperature, :Humidity, :Light, :CO2, :HumidityRatio, :Day, :Hour, :Minute, :Second, :Occupancy])
	
end

# ╔═╡ a7711e7b-288d-46c0-910d-51dc6cbc22e4
replace!(occupancy_date_df[!, "Occupancy"], 0 => -1.);

# ╔═╡ c036ff8b-267d-4673-8582-9b39ac4502ff
occupancy_date_train, occupancy_date_test = dataSplit(occupancy_date_df);

# ╔═╡ d2ebc549-3879-40c6-ad91-ccef317358ab
begin
	svm_occupancy_date = SVMRegular()
	fitSVM(svm_occupancy_date, occupancy_date_train)
end

# ╔═╡ 8701c933-08fb-48f1-bb62-cdcf94c4ba4f
resultsSVM(svm_occupancy_date, occupancy_date_test)

# ╔═╡ 6004e00c-3a31-429b-bfe7-4bce0b8a15bb
md"Comparing this to the results of the previus testing (without dates), we can see the accuracy changed in $(round(abs(92.3-96.1), digits=3))% in favor of this experiment using dates, so we can neglect them and use the original dataset without significant difference."

# ╔═╡ 14822bc5-0827-4c8b-91db-6d057fae8e33
md"# STOOOOP!!!!!!"

# ╔═╡ fdef7b51-3285-4388-ba09-91a4a6f22bd7
md"## Results
*Provide quantitative evidence for generalization using the provided datasets.*
- As explained in the SVM implementation while minimizing the norm $\mathbf{w}$ we are increasing the margin between the clusters of points. Therefore, wrong prediction will be less likely, as separation between classes is maximum. Also, when implementing regularization we are looking to minimize the empirical error of the sample, *i.e*, minimizing $\frac{1}{N}\sum_{i=1}^{N} L_{\mathbf{w}, b}(\mathbf{x}_I, y_i)$. This, added to the accuracy results gives us a percentage for generalization, as we divided data into training and testing, so we are **not** calculating the accuracy over the training data.

*Are these datasets linearly separable?*
- Accounting for noise in the datasets, we can say that datasets are linearly separable, as prediction using the SVM models yielded near 100% accuracy.

*Are these datasets randomly chosen?*
-

*Is the sample size enough to guarantee generalization?*
-
"

# ╔═╡ aaa21bda-3d18-4785-8166-f73be94a85eb
md"## References

1) YouTube. (2022, September 20). *How to implement SVM (Support Vector Machine) from scratch with Python*. Retrieved March 5, 2023, from <https://www.youtube.com/watch?v=T9UcK-TxQGw>.
"

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
CSV = "336ed68f-0bac-5ca0-87d4-7b16caf5d00b"
DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
Dates = "ade2ca70-3891-5945-98fb-dc099432e06a"
DecisionTree = "7806a523-6efd-50cb-b5f6-3fa6f1930dbb"
GLM = "38e38edf-8417-5370-95a0-9cbb8c7f171a"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
MLJ = "add582a8-e3ab-11e8-2d5e-e98b27df1bc7"
MLJLinearModels = "6ee0df7b-362f-4a72-a706-9e79364fb692"
PlotlyJS = "f0f68f2c-4968-5e81-91da-67840de0976a"
StatsPlots = "f3b207a7-027a-5e70-b257-86293d7955fd"

[compat]
CSV = "~0.10.9"
DataFrames = "~1.5.0"
DecisionTree = "~0.12.3"
GLM = "~1.8.1"
MLJ = "~0.19.1"
MLJLinearModels = "~0.9.1"
PlotlyJS = "~0.18.10"
StatsPlots = "~0.15.4"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.8.5"
manifest_format = "2.0"
project_hash = "b0b7c4ad1d5e257397349a578df9c2799bf32460"

[[deps.ARFFFiles]]
deps = ["CategoricalArrays", "Dates", "Parsers", "Tables"]
git-tree-sha1 = "e8c8e0a2be6eb4f56b1672e46004463033daa409"
uuid = "da404889-ca92-49ff-9e8b-0aa6b4d38dc8"
version = "1.4.1"

[[deps.AbstractFFTs]]
deps = ["ChainRulesCore", "LinearAlgebra"]
git-tree-sha1 = "69f7020bd72f069c219b5e8c236c1fa90d2cb409"
uuid = "621f4979-c628-5d54-868e-fcf4e3e8185c"
version = "1.2.1"

[[deps.AbstractTrees]]
git-tree-sha1 = "faa260e4cb5aba097a73fab382dd4b5819d8ec8c"
uuid = "1520ce14-60c1-5f80-bbc7-55ef81b5835c"
version = "0.4.4"

[[deps.Adapt]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "0310e08cb19f5da31d08341c6120c047598f5b9c"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "3.5.0"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.1"

[[deps.Arpack]]
deps = ["Arpack_jll", "Libdl", "LinearAlgebra", "Logging"]
git-tree-sha1 = "9b9b347613394885fd1c8c7729bfc60528faa436"
uuid = "7d9fca2a-8960-54d3-9f78-7d1dccf2cb97"
version = "0.5.4"

[[deps.Arpack_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "OpenBLAS_jll", "Pkg"]
git-tree-sha1 = "5ba6c757e8feccf03a1554dfaf3e26b3cfc7fd5e"
uuid = "68821587-b530-5797-8361-c406ea357684"
version = "3.5.1+1"

[[deps.ArrayInterface]]
deps = ["Adapt", "LinearAlgebra", "Requires", "SnoopPrecompile", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "4d9946e51e24f5e509779e3e2c06281a733914c2"
uuid = "4fba245c-0d91-5ea0-9b3e-6abc04ee57a9"
version = "7.1.0"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.AssetRegistry]]
deps = ["Distributed", "JSON", "Pidfile", "SHA", "Test"]
git-tree-sha1 = "b25e88db7944f98789130d7b503276bc34bc098e"
uuid = "bf4720bc-e11a-5d0c-854e-bdca1663c893"
version = "0.1.0"

[[deps.AxisAlgorithms]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "WoodburyMatrices"]
git-tree-sha1 = "66771c8d21c8ff5e3a93379480a2307ac36863f7"
uuid = "13072b0f-2c55-5437-9ae7-d433b7a33950"
version = "1.0.1"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[deps.BinDeps]]
deps = ["Libdl", "Pkg", "SHA", "URIParser", "Unicode"]
git-tree-sha1 = "1289b57e8cf019aede076edab0587eb9644175bd"
uuid = "9e28174c-4ba2-5203-b857-d8d62c4213ee"
version = "1.0.2"

[[deps.Blink]]
deps = ["Base64", "BinDeps", "Distributed", "JSExpr", "JSON", "Lazy", "Logging", "MacroTools", "Mustache", "Mux", "Reexport", "Sockets", "WebIO", "WebSockets"]
git-tree-sha1 = "08d0b679fd7caa49e2bca9214b131289e19808c0"
uuid = "ad839575-38b3-5650-b840-f874b8c74a25"
version = "0.12.5"

[[deps.Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "19a35467a82e236ff51bc17a3a44b69ef35185a2"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.8+0"

[[deps.CSV]]
deps = ["CodecZlib", "Dates", "FilePathsBase", "InlineStrings", "Mmap", "Parsers", "PooledArrays", "SentinelArrays", "SnoopPrecompile", "Tables", "Unicode", "WeakRefStrings", "WorkerUtilities"]
git-tree-sha1 = "c700cce799b51c9045473de751e9319bdd1c6e94"
uuid = "336ed68f-0bac-5ca0-87d4-7b16caf5d00b"
version = "0.10.9"

[[deps.Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "CompilerSupportLibraries_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "LZO_jll", "Libdl", "Pixman_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "4b859a208b2397a7a623a03449e4636bdb17bcf2"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.16.1+1"

[[deps.Calculus]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "f641eb0a4f00c343bbc32346e1217b86f3ce9dad"
uuid = "49dc2e85-a5d0-5ad3-a950-438e2897f1b9"
version = "0.5.1"

[[deps.CategoricalArrays]]
deps = ["DataAPI", "Future", "Missings", "Printf", "Requires", "Statistics", "Unicode"]
git-tree-sha1 = "5084cc1a28976dd1642c9f337b28a3cb03e0f7d2"
uuid = "324d7699-5711-5eae-9e2f-1d82baa6b597"
version = "0.10.7"

[[deps.CategoricalDistributions]]
deps = ["CategoricalArrays", "Distributions", "Missings", "OrderedCollections", "Random", "ScientificTypes", "UnicodePlots"]
git-tree-sha1 = "23fe4c6668776fedfd3747c545cd0d1a5190eb15"
uuid = "af321ab8-2d2e-40a6-b165-3d674595d28e"
version = "0.1.9"

[[deps.ChainRulesCore]]
deps = ["Compat", "LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "c6d890a52d2c4d55d326439580c3b8d0875a77d9"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.15.7"

[[deps.ChangesOfVariables]]
deps = ["ChainRulesCore", "LinearAlgebra", "Test"]
git-tree-sha1 = "485193efd2176b88e6622a39a246f8c5b600e74e"
uuid = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
version = "0.1.6"

[[deps.Clustering]]
deps = ["Distances", "LinearAlgebra", "NearestNeighbors", "Printf", "Random", "SparseArrays", "Statistics", "StatsBase"]
git-tree-sha1 = "64df3da1d2a26f4de23871cd1b6482bb68092bd5"
uuid = "aaaa29a8-35af-508c-8bc3-b662a17a0fe5"
version = "0.14.3"

[[deps.CodecZlib]]
deps = ["TranscodingStreams", "Zlib_jll"]
git-tree-sha1 = "9c209fb7536406834aa938fb149964b985de6c83"
uuid = "944b1d66-785c-5afd-91f1-9de20f533193"
version = "0.7.1"

[[deps.ColorSchemes]]
deps = ["ColorTypes", "ColorVectorSpace", "Colors", "FixedPointNumbers", "Random", "SnoopPrecompile"]
git-tree-sha1 = "aa3edc8f8dea6cbfa176ee12f7c2fc82f0608ed3"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.20.0"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "eb7f0f8307f71fac7c606984ea5fb2817275d6e4"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.4"

[[deps.ColorVectorSpace]]
deps = ["ColorTypes", "FixedPointNumbers", "LinearAlgebra", "SpecialFunctions", "Statistics", "TensorCore"]
git-tree-sha1 = "600cc5508d66b78aae350f7accdb58763ac18589"
uuid = "c3611d14-8923-5661-9e6a-0046d554d3a4"
version = "0.9.10"

[[deps.Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "fc08e5930ee9a4e03f84bfb5211cb54e7769758a"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.12.10"

[[deps.Combinatorics]]
git-tree-sha1 = "08c8b6831dc00bfea825826be0bc8336fc369860"
uuid = "861a8166-3701-5b0c-9a16-15d98fcdc6aa"
version = "1.0.2"

[[deps.CommonSubexpressions]]
deps = ["MacroTools", "Test"]
git-tree-sha1 = "7b8a93dba8af7e3b42fecabf646260105ac373f7"
uuid = "bbf7d656-a473-5ed7-a52c-81e309532950"
version = "0.3.0"

[[deps.Compat]]
deps = ["Dates", "LinearAlgebra", "UUIDs"]
git-tree-sha1 = "61fdd77467a5c3ad071ef8277ac6bd6af7dd4c04"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.6.0"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.0.1+0"

[[deps.ComputationalResources]]
git-tree-sha1 = "52cb3ec90e8a8bea0e62e275ba577ad0f74821f7"
uuid = "ed09eef8-17a6-5b46-8889-db040fac31e3"
version = "0.3.2"

[[deps.ConstructionBase]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "89a9db8d28102b094992472d333674bd1a83ce2a"
uuid = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
version = "1.5.1"

[[deps.Contour]]
git-tree-sha1 = "d05d9e7b7aedff4e5b51a029dced05cfb6125781"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.6.2"

[[deps.Crayons]]
git-tree-sha1 = "249fe38abf76d48563e2f4556bebd215aa317e15"
uuid = "a8cc5b0e-0ffa-5ad4-8c14-923d3ee1735f"
version = "4.1.1"

[[deps.DataAPI]]
git-tree-sha1 = "e8119c1a33d267e16108be441a287a6981ba1630"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.14.0"

[[deps.DataFrames]]
deps = ["Compat", "DataAPI", "Future", "InlineStrings", "InvertedIndices", "IteratorInterfaceExtensions", "LinearAlgebra", "Markdown", "Missings", "PooledArrays", "PrettyTables", "Printf", "REPL", "Random", "Reexport", "SentinelArrays", "SnoopPrecompile", "SortingAlgorithms", "Statistics", "TableTraits", "Tables", "Unicode"]
git-tree-sha1 = "aa51303df86f8626a962fccb878430cdb0a97eee"
uuid = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
version = "1.5.0"

[[deps.DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "d1fff3a548102f48987a52a2e0d114fa97d730f0"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.13"

[[deps.DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[deps.DataValues]]
deps = ["DataValueInterfaces", "Dates"]
git-tree-sha1 = "d88a19299eba280a6d062e135a43f00323ae70bf"
uuid = "e7dc6d0d-1eca-5fa6-8ad6-5aecde8b7ea5"
version = "0.4.13"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[deps.DecisionTree]]
deps = ["AbstractTrees", "DelimitedFiles", "LinearAlgebra", "Random", "ScikitLearnBase", "Statistics"]
git-tree-sha1 = "c6475a3ccad06cb1c2ebc0740c1bb4fe5a0731b7"
uuid = "7806a523-6efd-50cb-b5f6-3fa6f1930dbb"
version = "0.12.3"

[[deps.DelimitedFiles]]
deps = ["Mmap"]
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"

[[deps.DensityInterface]]
deps = ["InverseFunctions", "Test"]
git-tree-sha1 = "80c3e8639e3353e5d2912fb3a1916b8455e2494b"
uuid = "b429d917-457f-4dbc-8f4c-0cc954292b1d"
version = "0.4.0"

[[deps.DiffResults]]
deps = ["StaticArraysCore"]
git-tree-sha1 = "782dd5f4561f5d267313f23853baaaa4c52ea621"
uuid = "163ba53b-c6d8-5494-b064-1a9d43ac40c5"
version = "1.1.0"

[[deps.DiffRules]]
deps = ["IrrationalConstants", "LogExpFunctions", "NaNMath", "Random", "SpecialFunctions"]
git-tree-sha1 = "a4ad7ef19d2cdc2eff57abbbe68032b1cd0bd8f8"
uuid = "b552c78f-8df3-52c6-915a-8e097449b14b"
version = "1.13.0"

[[deps.Distances]]
deps = ["LinearAlgebra", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "3258d0659f812acde79e8a74b11f17ac06d0ca04"
uuid = "b4f34e82-e78d-54a5-968a-f98e89d6e8f7"
version = "0.10.7"

[[deps.Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"

[[deps.Distributions]]
deps = ["ChainRulesCore", "DensityInterface", "FillArrays", "LinearAlgebra", "PDMats", "Printf", "QuadGK", "Random", "SparseArrays", "SpecialFunctions", "Statistics", "StatsBase", "StatsFuns", "Test"]
git-tree-sha1 = "d71264a7b9a95dca3b8fff4477d94a837346c545"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.25.84"

[[deps.DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "2fb1e02f2b635d0845df5d7c167fec4dd739b00d"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.3"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.6.0"

[[deps.DualNumbers]]
deps = ["Calculus", "NaNMath", "SpecialFunctions"]
git-tree-sha1 = "5837a837389fccf076445fce071c8ddaea35a566"
uuid = "fa6b7ba4-c1ee-5f82-b5fc-ecf0adba8f74"
version = "0.6.8"

[[deps.EarlyStopping]]
deps = ["Dates", "Statistics"]
git-tree-sha1 = "98fdf08b707aaf69f524a6cd0a67858cefe0cfb6"
uuid = "792122b4-ca99-40de-a6bc-6742525f08b6"
version = "0.3.0"

[[deps.Expat_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "bad72f730e9e91c08d9427d5e8db95478a3c323d"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.4.8+0"

[[deps.FFMPEG]]
deps = ["FFMPEG_jll"]
git-tree-sha1 = "b57e3acbe22f8484b4b5ff66a7499717fe1a9cc8"
uuid = "c87230d0-a227-11e9-1b43-d7ebe4e7570a"
version = "0.4.1"

[[deps.FFMPEG_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "LAME_jll", "Libdl", "Ogg_jll", "OpenSSL_jll", "Opus_jll", "PCRE2_jll", "Pkg", "Zlib_jll", "libaom_jll", "libass_jll", "libfdk_aac_jll", "libvorbis_jll", "x264_jll", "x265_jll"]
git-tree-sha1 = "74faea50c1d007c85837327f6775bea60b5492dd"
uuid = "b22a6f82-2f65-5046-a5b2-351ab43fb4e5"
version = "4.4.2+2"

[[deps.FFTW]]
deps = ["AbstractFFTs", "FFTW_jll", "LinearAlgebra", "MKL_jll", "Preferences", "Reexport"]
git-tree-sha1 = "90630efff0894f8142308e334473eba54c433549"
uuid = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
version = "1.5.0"

[[deps.FFTW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c6033cc3892d0ef5bb9cd29b7f2f0331ea5184ea"
uuid = "f5851436-0d7a-5f13-b9de-f02708fd171a"
version = "3.3.10+0"

[[deps.FilePathsBase]]
deps = ["Compat", "Dates", "Mmap", "Printf", "Test", "UUIDs"]
git-tree-sha1 = "e27c4ebe80e8699540f2d6c805cc12203b614f12"
uuid = "48062228-2e41-5def-b9a4-89aafe57970f"
version = "0.9.20"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"

[[deps.FillArrays]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "Statistics"]
git-tree-sha1 = "d3ba08ab64bdfd27234d3f61956c966266757fe6"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "0.13.7"

[[deps.FiniteDiff]]
deps = ["ArrayInterface", "LinearAlgebra", "Requires", "Setfield", "SparseArrays", "StaticArrays"]
git-tree-sha1 = "ed1b56934a2f7a65035976985da71b6a65b4f2cf"
uuid = "6a86dc24-6348-571c-b903-95158fe2bd41"
version = "2.18.0"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "335bfdceacc84c5cdf16aadc768aa5ddfc5383cc"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.4"

[[deps.Fontconfig_jll]]
deps = ["Artifacts", "Bzip2_jll", "Expat_jll", "FreeType2_jll", "JLLWrappers", "Libdl", "Libuuid_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "21efd19106a55620a188615da6d3d06cd7f6ee03"
uuid = "a3f928ae-7b40-5064-980b-68af3947d34b"
version = "2.13.93+0"

[[deps.Formatting]]
deps = ["Printf"]
git-tree-sha1 = "8339d61043228fdd3eb658d86c926cb282ae72a8"
uuid = "59287772-0a20-5a39-b81b-1366585eb4c0"
version = "0.4.2"

[[deps.ForwardDiff]]
deps = ["CommonSubexpressions", "DiffResults", "DiffRules", "LinearAlgebra", "LogExpFunctions", "NaNMath", "Preferences", "Printf", "Random", "SpecialFunctions", "StaticArrays"]
git-tree-sha1 = "00e252f4d706b3d55a8863432e742bf5717b498d"
uuid = "f6369f11-7733-5829-9624-2563aa707210"
version = "0.10.35"

[[deps.FreeType2_jll]]
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "87eb71354d8ec1a96d4a7636bd57a7347dde3ef9"
uuid = "d7e528f0-a631-5988-bf34-fe36492bcfd7"
version = "2.10.4+0"

[[deps.FriBidi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "aa31987c2ba8704e23c6c8ba8a4f769d5d7e4f91"
uuid = "559328eb-81f9-559d-9380-de523a88c83c"
version = "1.0.10+0"

[[deps.FunctionalCollections]]
deps = ["Test"]
git-tree-sha1 = "04cb9cfaa6ba5311973994fe3496ddec19b6292a"
uuid = "de31a74c-ac4f-5751-b3fd-e18cd04993ca"
version = "0.5.0"

[[deps.Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"

[[deps.GLFW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libglvnd_jll", "Pkg", "Xorg_libXcursor_jll", "Xorg_libXi_jll", "Xorg_libXinerama_jll", "Xorg_libXrandr_jll"]
git-tree-sha1 = "d972031d28c8c8d9d7b41a536ad7bb0c2579caca"
uuid = "0656b61e-2033-5cc2-a64a-77c0f6c09b89"
version = "3.3.8+0"

[[deps.GLM]]
deps = ["Distributions", "LinearAlgebra", "Printf", "Reexport", "SparseArrays", "SpecialFunctions", "Statistics", "StatsAPI", "StatsBase", "StatsFuns", "StatsModels"]
git-tree-sha1 = "884477b9886a52a84378275737e2823a5c98e349"
uuid = "38e38edf-8417-5370-95a0-9cbb8c7f171a"
version = "1.8.1"

[[deps.GR]]
deps = ["Artifacts", "Base64", "DelimitedFiles", "Downloads", "GR_jll", "HTTP", "JSON", "Libdl", "LinearAlgebra", "Pkg", "Preferences", "Printf", "Random", "Serialization", "Sockets", "TOML", "Tar", "Test", "UUIDs", "p7zip_jll"]
git-tree-sha1 = "660b2ea2ec2b010bb02823c6d0ff6afd9bdc5c16"
uuid = "28b8d3ca-fb5f-59d9-8090-bfdbd6d07a71"
version = "0.71.7"

[[deps.GR_jll]]
deps = ["Artifacts", "Bzip2_jll", "Cairo_jll", "FFMPEG_jll", "Fontconfig_jll", "GLFW_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pixman_jll", "Qt5Base_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "d5e1fd17ac7f3aa4c5287a61ee28d4f8b8e98873"
uuid = "d2c73de3-f751-5644-a686-071e5b155ba9"
version = "0.71.7+0"

[[deps.Gettext_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "9b02998aba7bf074d14de89f9d37ca24a1a0b046"
uuid = "78b55507-aeef-58d4-861c-77aaff3498b1"
version = "0.21.0+0"

[[deps.Glib_jll]]
deps = ["Artifacts", "Gettext_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Libiconv_jll", "Libmount_jll", "PCRE2_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "d3b3624125c1474292d0d8ed0f65554ac37ddb23"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.74.0+2"

[[deps.Graphite2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "344bf40dcab1073aca04aa0df4fb092f920e4011"
uuid = "3b182d85-2403-5c21-9c21-1e1f0cc25472"
version = "1.3.14+0"

[[deps.Grisu]]
git-tree-sha1 = "53bb909d1151e57e2484c3d1b53e19552b887fb2"
uuid = "42e2da0e-8278-4e71-bc24-59509adca0fe"
version = "1.0.2"

[[deps.HTTP]]
deps = ["Base64", "Dates", "IniFile", "Logging", "MbedTLS", "NetworkOptions", "Sockets", "URIs"]
git-tree-sha1 = "0fa77022fe4b511826b39c894c90daf5fce3334a"
uuid = "cd3eb016-35fb-5094-929b-558a96fad6f3"
version = "0.9.17"

[[deps.HarfBuzz_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "Graphite2_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg"]
git-tree-sha1 = "129acf094d168394e80ee1dc4bc06ec835e510a3"
uuid = "2e76f6c2-a576-52d4-95c1-20adfe4de566"
version = "2.8.1+1"

[[deps.Hiccup]]
deps = ["MacroTools", "Test"]
git-tree-sha1 = "6187bb2d5fcbb2007c39e7ac53308b0d371124bd"
uuid = "9fb69e20-1954-56bb-a84f-559cc56a8ff7"
version = "0.2.2"

[[deps.HypergeometricFunctions]]
deps = ["DualNumbers", "LinearAlgebra", "OpenLibm_jll", "SpecialFunctions", "Test"]
git-tree-sha1 = "709d864e3ed6e3545230601f94e11ebc65994641"
uuid = "34004b35-14d8-5ef3-9330-4cdb6864b03a"
version = "0.3.11"

[[deps.IniFile]]
git-tree-sha1 = "f550e6e32074c939295eb5ea6de31849ac2c9625"
uuid = "83e8ac13-25f8-5344-8a64-a9f2b223428f"
version = "0.5.1"

[[deps.InlineStrings]]
deps = ["Parsers"]
git-tree-sha1 = "9cc2baf75c6d09f9da536ddf58eb2f29dedaf461"
uuid = "842dd82b-1e85-43dc-bf29-5d0ee9dffc48"
version = "1.4.0"

[[deps.IntelOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "d979e54b71da82f3a65b62553da4fc3d18c9004c"
uuid = "1d5cc7b8-4909-519e-a0f8-d0f5ad9712d0"
version = "2018.0.3+2"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[deps.Interpolations]]
deps = ["Adapt", "AxisAlgorithms", "ChainRulesCore", "LinearAlgebra", "OffsetArrays", "Random", "Ratios", "Requires", "SharedArrays", "SparseArrays", "StaticArrays", "WoodburyMatrices"]
git-tree-sha1 = "721ec2cf720536ad005cb38f50dbba7b02419a15"
uuid = "a98d9a8b-a2ab-59e6-89dd-64a1c18fca59"
version = "0.14.7"

[[deps.InverseFunctions]]
deps = ["Test"]
git-tree-sha1 = "49510dfcb407e572524ba94aeae2fced1f3feb0f"
uuid = "3587e190-3f89-42d0-90ee-14403ec27112"
version = "0.1.8"

[[deps.InvertedIndices]]
git-tree-sha1 = "82aec7a3dd64f4d9584659dc0b62ef7db2ef3e19"
uuid = "41ab1584-1d38-5bbf-9106-f11c6c58b48f"
version = "1.2.0"

[[deps.IrrationalConstants]]
git-tree-sha1 = "3868cac300a188a7c3a74f9abd930e52ce1a7a51"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.2.1"

[[deps.IterationControl]]
deps = ["EarlyStopping", "InteractiveUtils"]
git-tree-sha1 = "d7df9a6fdd82a8cfdfe93a94fcce35515be634da"
uuid = "b3c1a2ee-3fec-4384-bf48-272ea71de57c"
version = "0.5.3"

[[deps.IterativeSolvers]]
deps = ["LinearAlgebra", "Printf", "Random", "RecipesBase", "SparseArrays"]
git-tree-sha1 = "1169632f425f79429f245113b775a0e3d121457c"
uuid = "42fd0dbc-a981-5370-80f2-aaf504508153"
version = "0.9.2"

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[deps.JLFzf]]
deps = ["Pipe", "REPL", "Random", "fzf_jll"]
git-tree-sha1 = "f377670cda23b6b7c1c0b3893e37451c5c1a2185"
uuid = "1019f520-868f-41f5-a6de-eb00f4b6a39c"
version = "0.1.5"

[[deps.JLLWrappers]]
deps = ["Preferences"]
git-tree-sha1 = "abc9885a7ca2052a736a600f7fa66209f96506e1"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.4.1"

[[deps.JSExpr]]
deps = ["JSON", "MacroTools", "Observables", "WebIO"]
git-tree-sha1 = "b413a73785b98474d8af24fd4c8a975e31df3658"
uuid = "97c1335a-c9c5-57fe-bc5d-ec35cebe8660"
version = "0.5.4"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "3c837543ddb02250ef42f4738347454f95079d4e"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.3"

[[deps.JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "6f2675ef130a300a112286de91973805fcc5ffbc"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "2.1.91+0"

[[deps.Kaleido_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "43032da5832754f58d14a91ffbe86d5f176acda9"
uuid = "f7e6163d-2fa5-5f23-b69c-1db539e41963"
version = "0.2.1+0"

[[deps.KernelDensity]]
deps = ["Distributions", "DocStringExtensions", "FFTW", "Interpolations", "StatsBase"]
git-tree-sha1 = "9816b296736292a80b9a3200eb7fbb57aaa3917a"
uuid = "5ab0869b-81aa-558d-bb23-cbf5423bbe9b"
version = "0.6.5"

[[deps.LAME_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "f6250b16881adf048549549fba48b1161acdac8c"
uuid = "c1c5ebd0-6772-5130-a774-d5fcae4a789d"
version = "3.100.1+0"

[[deps.LERC_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "bf36f528eec6634efc60d7ec062008f171071434"
uuid = "88015f11-f218-50d7-93a8-a6af411a945d"
version = "3.0.0+1"

[[deps.LZO_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e5b909bcf985c5e2605737d2ce278ed791b89be6"
uuid = "dd4b983a-f0e5-5f8d-a1b7-129d4a5fb1ac"
version = "2.10.1+0"

[[deps.LaTeXStrings]]
git-tree-sha1 = "f2355693d6778a178ade15952b7ac47a4ff97996"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.3.0"

[[deps.Latexify]]
deps = ["Formatting", "InteractiveUtils", "LaTeXStrings", "MacroTools", "Markdown", "OrderedCollections", "Printf", "Requires"]
git-tree-sha1 = "2422f47b34d4b127720a18f86fa7b1aa2e141f29"
uuid = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
version = "0.15.18"

[[deps.LatinHypercubeSampling]]
deps = ["Random", "StableRNGs", "StatsBase", "Test"]
git-tree-sha1 = "42938ab65e9ed3c3029a8d2c58382ca75bdab243"
uuid = "a5e1c1ea-c99a-51d3-a14d-a9a37257b02d"
version = "1.8.0"

[[deps.Lazy]]
deps = ["MacroTools"]
git-tree-sha1 = "1370f8202dac30758f3c345f9909b97f53d87d3f"
uuid = "50d2b5c4-7a5e-59d5-8109-a42b560f39c0"
version = "0.15.1"

[[deps.LazyArtifacts]]
deps = ["Artifacts", "Pkg"]
uuid = "4af54fe1-eca0-43a8-85a7-787d91b784e3"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.3"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "7.84.0+0"

[[deps.LibGit2]]
deps = ["Base64", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.10.2+0"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[deps.Libffi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "0b4a5d71f3e5200a7dff793393e09dfc2d874290"
uuid = "e9f186c6-92d2-5b65-8a66-fee21dc1b490"
version = "3.2.2+1"

[[deps.Libgcrypt_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgpg_error_jll", "Pkg"]
git-tree-sha1 = "64613c82a59c120435c067c2b809fc61cf5166ae"
uuid = "d4300ac3-e22c-5743-9152-c294e39db1e4"
version = "1.8.7+0"

[[deps.Libglvnd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll", "Xorg_libXext_jll"]
git-tree-sha1 = "6f73d1dd803986947b2c750138528a999a6c7733"
uuid = "7e76a0d4-f3c7-5321-8279-8d96eeed0f29"
version = "1.6.0+0"

[[deps.Libgpg_error_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c333716e46366857753e273ce6a69ee0945a6db9"
uuid = "7add5ba3-2f88-524e-9cd5-f83b8a55f7b8"
version = "1.42.0+0"

[[deps.Libiconv_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c7cb1f5d892775ba13767a87c7ada0b980ea0a71"
uuid = "94ce4f54-9a6c-5748-9c1c-f9c7231a4531"
version = "1.16.1+2"

[[deps.Libmount_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "9c30530bf0effd46e15e0fdcf2b8636e78cbbd73"
uuid = "4b2f31a3-9ecc-558c-b454-b3730dcb73e9"
version = "2.35.0+0"

[[deps.Libtiff_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "LERC_jll", "Libdl", "Pkg", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "3eb79b0ca5764d4799c06699573fd8f533259713"
uuid = "89763e89-9b03-5906-acba-b20f662cd828"
version = "4.4.0+0"

[[deps.Libuuid_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "7f3efec06033682db852f8b3bc3c1d2b0a0ab066"
uuid = "38a345b3-de98-5d2b-a5d3-14cd9215e700"
version = "2.36.0+0"

[[deps.LineSearches]]
deps = ["LinearAlgebra", "NLSolversBase", "NaNMath", "Parameters", "Printf"]
git-tree-sha1 = "7bbea35cec17305fc70a0e5b4641477dc0789d9d"
uuid = "d3d80556-e9d4-5f37-9878-2ab0fcc64255"
version = "7.2.0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[deps.LinearMaps]]
deps = ["ChainRulesCore", "LinearAlgebra", "SparseArrays", "Statistics"]
git-tree-sha1 = "42970dad6b0d2515571613010bd32ba37e07f874"
uuid = "7a12625a-238d-50fd-b39a-03d52299707e"
version = "3.9.0"

[[deps.LogExpFunctions]]
deps = ["ChainRulesCore", "ChangesOfVariables", "DocStringExtensions", "InverseFunctions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "0a1b7c2863e44523180fdb3146534e265a91870b"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.23"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[deps.LossFunctions]]
deps = ["InteractiveUtils", "Markdown", "RecipesBase"]
git-tree-sha1 = "53cd63a12f06a43eef6f4aafb910ac755c122be7"
uuid = "30fc2ffe-d236-52d8-8643-a9d8f7c094a7"
version = "0.8.0"

[[deps.MKL_jll]]
deps = ["Artifacts", "IntelOpenMP_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "Pkg"]
git-tree-sha1 = "2ce8695e1e699b68702c03402672a69f54b8aca9"
uuid = "856f044c-d86e-5d09-b602-aeab76dc8ba7"
version = "2022.2.0+0"

[[deps.MLJ]]
deps = ["CategoricalArrays", "ComputationalResources", "Distributed", "Distributions", "LinearAlgebra", "MLJBase", "MLJEnsembles", "MLJIteration", "MLJModels", "MLJTuning", "OpenML", "Pkg", "ProgressMeter", "Random", "ScientificTypes", "Statistics", "StatsBase", "Tables"]
git-tree-sha1 = "80149328ca780b522b5a95e402450d10df7904f2"
uuid = "add582a8-e3ab-11e8-2d5e-e98b27df1bc7"
version = "0.19.1"

[[deps.MLJBase]]
deps = ["CategoricalArrays", "CategoricalDistributions", "ComputationalResources", "Dates", "DelimitedFiles", "Distributed", "Distributions", "InteractiveUtils", "InvertedIndices", "LinearAlgebra", "LossFunctions", "MLJModelInterface", "Missings", "OrderedCollections", "Parameters", "PrettyTables", "ProgressMeter", "Random", "ScientificTypes", "Serialization", "StatisticalTraits", "Statistics", "StatsBase", "Tables"]
git-tree-sha1 = "f6667db64f84c5031e3f4e48b5da80e1dd39429d"
uuid = "a7f614a8-145f-11e9-1d2a-a57a1082229d"
version = "0.21.5"

[[deps.MLJEnsembles]]
deps = ["CategoricalArrays", "CategoricalDistributions", "ComputationalResources", "Distributed", "Distributions", "MLJBase", "MLJModelInterface", "ProgressMeter", "Random", "ScientificTypesBase", "StatsBase"]
git-tree-sha1 = "bb8a1056b1d8b40f2f27167fc3ef6412a6719fbf"
uuid = "50ed68f4-41fd-4504-931a-ed422449fee0"
version = "0.3.2"

[[deps.MLJIteration]]
deps = ["IterationControl", "MLJBase", "Random", "Serialization"]
git-tree-sha1 = "be6d5c71ab499a59e82d65e00a89ceba8732fcd5"
uuid = "614be32b-d00c-4edb-bd02-1eb411ab5e55"
version = "0.5.1"

[[deps.MLJLinearModels]]
deps = ["DocStringExtensions", "IterativeSolvers", "LinearAlgebra", "LinearMaps", "MLJModelInterface", "Optim", "Parameters"]
git-tree-sha1 = "c811b3877f1328179cef6662388d200c78b95c09"
uuid = "6ee0df7b-362f-4a72-a706-9e79364fb692"
version = "0.9.1"

[[deps.MLJModelInterface]]
deps = ["Random", "ScientificTypesBase", "StatisticalTraits"]
git-tree-sha1 = "c8b7e632d6754a5e36c0d94a4b466a5ba3a30128"
uuid = "e80e1ace-859a-464e-9ed9-23947d8ae3ea"
version = "1.8.0"

[[deps.MLJModels]]
deps = ["CategoricalArrays", "CategoricalDistributions", "Combinatorics", "Dates", "Distances", "Distributions", "InteractiveUtils", "LinearAlgebra", "MLJModelInterface", "Markdown", "OrderedCollections", "Parameters", "Pkg", "PrettyPrinting", "REPL", "Random", "RelocatableFolders", "ScientificTypes", "StatisticalTraits", "Statistics", "StatsBase", "Tables"]
git-tree-sha1 = "1d445497ca058dbc0dbc7528b778707893edb969"
uuid = "d491faf4-2d78-11e9-2867-c94bc002c0b7"
version = "0.16.4"

[[deps.MLJTuning]]
deps = ["ComputationalResources", "Distributed", "Distributions", "LatinHypercubeSampling", "MLJBase", "ProgressMeter", "Random", "RecipesBase"]
git-tree-sha1 = "02688098bd77827b64ed8ad747c14f715f98cfc4"
uuid = "03970b2e-30c4-11ea-3135-d1576263f10f"
version = "0.7.4"

[[deps.MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "42324d08725e200c23d4dfb549e0d5d89dede2d2"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.10"

[[deps.MarchingCubes]]
deps = ["SnoopPrecompile", "StaticArrays"]
git-tree-sha1 = "55aaf3fdf414b691a15875cfe5edb6e0daf4625a"
uuid = "299715c1-40a9-479a-aaf9-4a633d36f717"
version = "0.1.6"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[deps.MbedTLS]]
deps = ["Dates", "MbedTLS_jll", "MozillaCACerts_jll", "Random", "Sockets"]
git-tree-sha1 = "03a9b9718f5682ecb107ac9f7308991db4ce395b"
uuid = "739be429-bea8-5141-9913-cc70e7f3736d"
version = "1.1.7"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.0+0"

[[deps.Measures]]
git-tree-sha1 = "c13304c81eec1ed3af7fc20e75fb6b26092a1102"
uuid = "442fdcdd-2543-5da2-b0f3-8c86c306513e"
version = "0.3.2"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "f66bdc5de519e8f8ae43bdc598782d35a25b1272"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.1.0"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2022.2.1"

[[deps.MultivariateStats]]
deps = ["Arpack", "LinearAlgebra", "SparseArrays", "Statistics", "StatsAPI", "StatsBase"]
git-tree-sha1 = "91a48569383df24f0fd2baf789df2aade3d0ad80"
uuid = "6f286f6a-111f-5878-ab1e-185364afe411"
version = "0.10.1"

[[deps.Mustache]]
deps = ["Printf", "Tables"]
git-tree-sha1 = "1e566ae913a57d0062ff1af54d2697b9344b99cd"
uuid = "ffc61752-8dc7-55ee-8c37-f3e9cdd09e70"
version = "1.0.14"

[[deps.Mux]]
deps = ["AssetRegistry", "Base64", "HTTP", "Hiccup", "Pkg", "Sockets", "WebSockets"]
git-tree-sha1 = "82dfb2cead9895e10ee1b0ca37a01088456c4364"
uuid = "a975b10e-0019-58db-a62f-e48ff68538c9"
version = "0.7.6"

[[deps.NLSolversBase]]
deps = ["DiffResults", "Distributed", "FiniteDiff", "ForwardDiff"]
git-tree-sha1 = "a0b464d183da839699f4c79e7606d9d186ec172c"
uuid = "d41bc354-129a-5804-8e4c-c37616107c6c"
version = "7.8.3"

[[deps.NaNMath]]
deps = ["OpenLibm_jll"]
git-tree-sha1 = "0877504529a3e5c3343c6f8b4c0381e57e4387e4"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.0.2"

[[deps.NearestNeighbors]]
deps = ["Distances", "StaticArrays"]
git-tree-sha1 = "2c3726ceb3388917602169bed973dbc97f1b51a8"
uuid = "b8a86587-4115-5ab1-83bc-aa920d37bbce"
version = "0.4.13"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.2.0"

[[deps.Observables]]
git-tree-sha1 = "6862738f9796b3edc1c09d0890afce4eca9e7e93"
uuid = "510215fc-4207-5dde-b226-833fc4488ee2"
version = "0.5.4"

[[deps.OffsetArrays]]
deps = ["Adapt"]
git-tree-sha1 = "82d7c9e310fe55aa54996e6f7f94674e2a38fcb4"
uuid = "6fe1bfb0-de20-5000-8ca7-80f57d26f881"
version = "1.12.9"

[[deps.Ogg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "887579a3eb005446d514ab7aeac5d1d027658b8f"
uuid = "e7412a2a-1a6e-54c0-be00-318e2571c051"
version = "1.3.5+1"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.20+0"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"
version = "0.8.1+0"

[[deps.OpenML]]
deps = ["ARFFFiles", "HTTP", "JSON", "Markdown", "Pkg", "Scratch"]
git-tree-sha1 = "6efb039ae888699d5a74fb593f6f3e10c7193e33"
uuid = "8b6db2d4-7670-4922-a472-f9537c81ab66"
version = "0.3.1"

[[deps.OpenSSL_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "9ff31d101d987eb9d66bd8b176ac7c277beccd09"
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "1.1.20+0"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "13652491f6856acfd2db29360e1bbcd4565d04f1"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.5+0"

[[deps.Optim]]
deps = ["Compat", "FillArrays", "ForwardDiff", "LineSearches", "LinearAlgebra", "NLSolversBase", "NaNMath", "Parameters", "PositiveFactorizations", "Printf", "SparseArrays", "StatsBase"]
git-tree-sha1 = "1903afc76b7d01719d9c30d3c7d501b61db96721"
uuid = "429524aa-4258-5aef-a3af-852621145aeb"
version = "1.7.4"

[[deps.Opus_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "51a08fb14ec28da2ec7a927c4337e4332c2a4720"
uuid = "91d4177d-7536-5919-b921-800302f37372"
version = "1.3.2+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "85f8e6578bf1f9ee0d11e7bb1b1456435479d47c"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.4.1"

[[deps.PCRE2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "efcefdf7-47ab-520b-bdef-62a2eaa19f15"
version = "10.40.0+0"

[[deps.PDMats]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "cf494dca75a69712a72b80bc48f59dcf3dea63ec"
uuid = "90014a1f-27ba-587c-ab20-58faa44d9150"
version = "0.11.16"

[[deps.Parameters]]
deps = ["OrderedCollections", "UnPack"]
git-tree-sha1 = "34c0e9ad262e5f7fc75b10a9952ca7692cfc5fbe"
uuid = "d96e819e-fc66-5662-9728-84c9c7592b0a"
version = "0.12.3"

[[deps.Parsers]]
deps = ["Dates", "SnoopPrecompile"]
git-tree-sha1 = "6f4fbcd1ad45905a5dee3f4256fabb49aa2110c6"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.5.7"

[[deps.Pidfile]]
deps = ["FileWatching", "Test"]
git-tree-sha1 = "2d8aaf8ee10df53d0dfb9b8ee44ae7c04ced2b03"
uuid = "fa939f87-e72e-5be4-a000-7fc836dbe307"
version = "1.3.0"

[[deps.Pipe]]
git-tree-sha1 = "6842804e7867b115ca9de748a0cf6b364523c16d"
uuid = "b98c9c47-44ae-5843-9183-064241ee97a0"
version = "1.3.0"

[[deps.Pixman_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b4f5d02549a10e20780a24fce72bea96b6329e29"
uuid = "30392449-352a-5448-841d-b1acce4e97dc"
version = "0.40.1+0"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.8.0"

[[deps.PlotThemes]]
deps = ["PlotUtils", "Statistics"]
git-tree-sha1 = "1f03a2d339f42dca4a4da149c7e15e9b896ad899"
uuid = "ccf2f8ad-2431-5c83-bf29-c5338b663b6a"
version = "3.1.0"

[[deps.PlotUtils]]
deps = ["ColorSchemes", "Colors", "Dates", "Printf", "Random", "Reexport", "SnoopPrecompile", "Statistics"]
git-tree-sha1 = "c95373e73290cf50a8a22c3375e4625ded5c5280"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.3.4"

[[deps.PlotlyBase]]
deps = ["ColorSchemes", "Dates", "DelimitedFiles", "DocStringExtensions", "JSON", "LaTeXStrings", "Logging", "Parameters", "Pkg", "REPL", "Requires", "Statistics", "UUIDs"]
git-tree-sha1 = "56baf69781fc5e61607c3e46227ab17f7040ffa2"
uuid = "a03496cd-edff-5a9b-9e67-9cda94a718b5"
version = "0.8.19"

[[deps.PlotlyJS]]
deps = ["Base64", "Blink", "DelimitedFiles", "JSExpr", "JSON", "Kaleido_jll", "Markdown", "Pkg", "PlotlyBase", "REPL", "Reexport", "Requires", "WebIO"]
git-tree-sha1 = "7452869933cd5af22f59557390674e8679ab2338"
uuid = "f0f68f2c-4968-5e81-91da-67840de0976a"
version = "0.18.10"

[[deps.Plots]]
deps = ["Base64", "Contour", "Dates", "Downloads", "FFMPEG", "FixedPointNumbers", "GR", "JLFzf", "JSON", "LaTeXStrings", "Latexify", "LinearAlgebra", "Measures", "NaNMath", "Pkg", "PlotThemes", "PlotUtils", "Preferences", "Printf", "REPL", "Random", "RecipesBase", "RecipesPipeline", "Reexport", "RelocatableFolders", "Requires", "Scratch", "Showoff", "SnoopPrecompile", "SparseArrays", "Statistics", "StatsBase", "UUIDs", "UnicodeFun", "Unzip"]
git-tree-sha1 = "da1d3fb7183e38603fcdd2061c47979d91202c97"
uuid = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
version = "1.38.6"

[[deps.PooledArrays]]
deps = ["DataAPI", "Future"]
git-tree-sha1 = "a6062fe4063cdafe78f4a0a81cfffb89721b30e7"
uuid = "2dfb63ee-cc39-5dd5-95bd-886bf059d720"
version = "1.4.2"

[[deps.PositiveFactorizations]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "17275485f373e6673f7e7f97051f703ed5b15b20"
uuid = "85a6dd25-e78a-55b7-8502-1745935b8125"
version = "0.2.4"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "47e5f437cc0e7ef2ce8406ce1e7e24d44915f88d"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.3.0"

[[deps.PrettyPrinting]]
git-tree-sha1 = "4be53d093e9e37772cc89e1009e8f6ad10c4681b"
uuid = "54e16d92-306c-5ea0-a30b-337be88ac337"
version = "0.4.0"

[[deps.PrettyTables]]
deps = ["Crayons", "Formatting", "LaTeXStrings", "Markdown", "Reexport", "StringManipulation", "Tables"]
git-tree-sha1 = "96f6db03ab535bdb901300f88335257b0018689d"
uuid = "08abe8d2-0d0c-5749-adfa-8a2ac140af0d"
version = "2.2.2"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.ProgressMeter]]
deps = ["Distributed", "Printf"]
git-tree-sha1 = "d7a7aef8f8f2d537104f170139553b14dfe39fe9"
uuid = "92933f4c-e287-5a05-a399-4b506db050ca"
version = "1.7.2"

[[deps.Qt5Base_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Fontconfig_jll", "Glib_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "OpenSSL_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libxcb_jll", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_keysyms_jll", "Xorg_xcb_util_renderutil_jll", "Xorg_xcb_util_wm_jll", "Zlib_jll", "xkbcommon_jll"]
git-tree-sha1 = "0c03844e2231e12fda4d0086fd7cbe4098ee8dc5"
uuid = "ea2cea3b-5b76-57ae-a6ef-0a8af62496e1"
version = "5.15.3+2"

[[deps.QuadGK]]
deps = ["DataStructures", "LinearAlgebra"]
git-tree-sha1 = "786efa36b7eff813723c4849c90456609cf06661"
uuid = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
version = "2.8.1"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.Random]]
deps = ["SHA", "Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[deps.Ratios]]
deps = ["Requires"]
git-tree-sha1 = "dc84268fe0e3335a62e315a3a7cf2afa7178a734"
uuid = "c84ed2f1-dad5-54f0-aa8e-dbefe2724439"
version = "0.4.3"

[[deps.RecipesBase]]
deps = ["SnoopPrecompile"]
git-tree-sha1 = "261dddd3b862bd2c940cf6ca4d1c8fe593e457c8"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.3.3"

[[deps.RecipesPipeline]]
deps = ["Dates", "NaNMath", "PlotUtils", "RecipesBase", "SnoopPrecompile"]
git-tree-sha1 = "e974477be88cb5e3040009f3767611bc6357846f"
uuid = "01d81517-befc-4cb6-b9ec-a95719d0359c"
version = "0.6.11"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.RelocatableFolders]]
deps = ["SHA", "Scratch"]
git-tree-sha1 = "90bc7a7c96410424509e4263e277e43250c05691"
uuid = "05181044-ff0b-4ac5-8273-598c1e38db00"
version = "1.0.0"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "838a3a4188e2ded87a4f9f184b4b0d78a1e91cb7"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.0"

[[deps.Rmath]]
deps = ["Random", "Rmath_jll"]
git-tree-sha1 = "f65dcb5fa46aee0cf9ed6274ccbd597adc49aa7b"
uuid = "79098fc4-a85e-5d69-aa6a-4863f24498fa"
version = "0.7.1"

[[deps.Rmath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "6ed52fdd3382cf21947b15e8870ac0ddbff736da"
uuid = "f50d1b31-88e8-58de-be2c-1cc44531875f"
version = "0.4.0+0"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.ScientificTypes]]
deps = ["CategoricalArrays", "ColorTypes", "Dates", "Distributions", "PrettyTables", "Reexport", "ScientificTypesBase", "StatisticalTraits", "Tables"]
git-tree-sha1 = "75ccd10ca65b939dab03b812994e571bf1e3e1da"
uuid = "321657f4-b219-11e9-178b-2701a2544e81"
version = "3.0.2"

[[deps.ScientificTypesBase]]
git-tree-sha1 = "a8e18eb383b5ecf1b5e6fc237eb39255044fd92b"
uuid = "30f210dd-8aff-4c5f-94ba-8e64358c1161"
version = "3.0.0"

[[deps.ScikitLearnBase]]
deps = ["LinearAlgebra", "Random", "Statistics"]
git-tree-sha1 = "7877e55c1523a4b336b433da39c8e8c08d2f221f"
uuid = "6e75b9c4-186b-50bd-896f-2d2496a4843e"
version = "0.5.0"

[[deps.Scratch]]
deps = ["Dates"]
git-tree-sha1 = "f94f779c94e58bf9ea243e77a37e16d9de9126bd"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.1.1"

[[deps.SentinelArrays]]
deps = ["Dates", "Random"]
git-tree-sha1 = "77d3c4726515dca71f6d80fbb5e251088defe305"
uuid = "91c51154-3ec4-41a3-a24f-3f23e20d615c"
version = "1.3.18"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[deps.Setfield]]
deps = ["ConstructionBase", "Future", "MacroTools", "StaticArraysCore"]
git-tree-sha1 = "e2cc6d8c88613c05e1defb55170bf5ff211fbeac"
uuid = "efcf1570-3423-57d1-acb7-fd33fddbac46"
version = "1.1.1"

[[deps.SharedArrays]]
deps = ["Distributed", "Mmap", "Random", "Serialization"]
uuid = "1a1011a3-84de-559e-8e89-a11a2f7dc383"

[[deps.ShiftedArrays]]
git-tree-sha1 = "503688b59397b3307443af35cd953a13e8005c16"
uuid = "1277b4bf-5013-50f5-be3d-901d8477a67a"
version = "2.0.0"

[[deps.Showoff]]
deps = ["Dates", "Grisu"]
git-tree-sha1 = "91eddf657aca81df9ae6ceb20b959ae5653ad1de"
uuid = "992d4aef-0814-514b-bc4d-f2e9a6c4116f"
version = "1.0.3"

[[deps.SnoopPrecompile]]
deps = ["Preferences"]
git-tree-sha1 = "e760a70afdcd461cf01a575947738d359234665c"
uuid = "66db9d55-30c0-4569-8b51-7e840670fc0c"
version = "1.0.3"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "a4ada03f999bd01b3a25dcaa30b2d929fe537e00"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.1.0"

[[deps.SparseArrays]]
deps = ["LinearAlgebra", "Random"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.SpecialFunctions]]
deps = ["ChainRulesCore", "IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "ef28127915f4229c971eb43f3fc075dd3fe91880"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.2.0"

[[deps.StableRNGs]]
deps = ["Random", "Test"]
git-tree-sha1 = "3be7d49667040add7ee151fefaf1f8c04c8c8276"
uuid = "860ef19b-820b-49d6-a774-d7a799459cd3"
version = "1.0.0"

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "Random", "StaticArraysCore", "Statistics"]
git-tree-sha1 = "2d7d9e1ddadc8407ffd460e24218e37ef52dd9a3"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.5.16"

[[deps.StaticArraysCore]]
git-tree-sha1 = "6b7ba252635a5eff6a0b0664a41ee140a1c9e72a"
uuid = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
version = "1.4.0"

[[deps.StatisticalTraits]]
deps = ["ScientificTypesBase"]
git-tree-sha1 = "30b9236691858e13f167ce829490a68e1a597782"
uuid = "64bff920-2084-43da-a3e6-9bb72801c0c9"
version = "3.2.0"

[[deps.Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[deps.StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "f9af7f195fb13589dd2e2d57fdb401717d2eb1f6"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.5.0"

[[deps.StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "d1bf48bfcc554a3761a133fe3a9bb01488e06916"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.33.21"

[[deps.StatsFuns]]
deps = ["ChainRulesCore", "HypergeometricFunctions", "InverseFunctions", "IrrationalConstants", "LogExpFunctions", "Reexport", "Rmath", "SpecialFunctions"]
git-tree-sha1 = "5aa6250a781e567388f3285fb4b0f214a501b4d5"
uuid = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
version = "1.2.1"

[[deps.StatsModels]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "Printf", "REPL", "ShiftedArrays", "SparseArrays", "StatsBase", "StatsFuns", "Tables"]
git-tree-sha1 = "a5e15f27abd2692ccb61a99e0854dfb7d48017db"
uuid = "3eaba693-59b7-5ba5-a881-562e759f1c8d"
version = "0.6.33"

[[deps.StatsPlots]]
deps = ["AbstractFFTs", "Clustering", "DataStructures", "DataValues", "Distributions", "Interpolations", "KernelDensity", "LinearAlgebra", "MultivariateStats", "NaNMath", "Observables", "Plots", "RecipesBase", "RecipesPipeline", "Reexport", "StatsBase", "TableOperations", "Tables", "Widgets"]
git-tree-sha1 = "e0d5bc26226ab1b7648278169858adcfbd861780"
uuid = "f3b207a7-027a-5e70-b257-86293d7955fd"
version = "0.15.4"

[[deps.StringManipulation]]
git-tree-sha1 = "46da2434b41f41ac3594ee9816ce5541c6096123"
uuid = "892a3eda-7b42-436c-8928-eab12a02cf0e"
version = "0.3.0"

[[deps.SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.0"

[[deps.TableOperations]]
deps = ["SentinelArrays", "Tables", "Test"]
git-tree-sha1 = "e383c87cf2a1dc41fa30c093b2a19877c83e1bc1"
uuid = "ab02a1b2-a7df-11e8-156e-fb1833f50b87"
version = "1.2.0"

[[deps.TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[deps.Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "LinearAlgebra", "OrderedCollections", "TableTraits", "Test"]
git-tree-sha1 = "c79322d36826aa2f4fd8ecfa96ddb47b174ac78d"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.10.0"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.1"

[[deps.TensorCore]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1feb45f88d133a655e001435632f019a9a1bcdb6"
uuid = "62fd8b95-f654-4bbd-a8a5-9c27f68ccd50"
version = "0.1.1"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.TranscodingStreams]]
deps = ["Random", "Test"]
git-tree-sha1 = "94f38103c984f89cf77c402f2a68dbd870f8165f"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.9.11"

[[deps.URIParser]]
deps = ["Unicode"]
git-tree-sha1 = "53a9f49546b8d2dd2e688d216421d050c9a31d0d"
uuid = "30578b45-9adc-5946-b283-645ec420af67"
version = "0.4.1"

[[deps.URIs]]
git-tree-sha1 = "074f993b0ca030848b897beff716d93aca60f06a"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.4.2"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[deps.UnPack]]
git-tree-sha1 = "387c1f73762231e86e0c9c5443ce3b4a0a9a0c2b"
uuid = "3a884ed6-31ef-47d7-9d2a-63182c4928ed"
version = "1.0.2"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[deps.UnicodeFun]]
deps = ["REPL"]
git-tree-sha1 = "53915e50200959667e78a92a418594b428dffddf"
uuid = "1cfade01-22cf-5700-b092-accc4b62d6e1"
version = "0.4.1"

[[deps.UnicodePlots]]
deps = ["ColorSchemes", "ColorTypes", "Contour", "Crayons", "Dates", "LinearAlgebra", "MarchingCubes", "NaNMath", "Printf", "Requires", "SnoopPrecompile", "SparseArrays", "StaticArrays", "StatsBase"]
git-tree-sha1 = "ef00b38d086414a54d679d81ced90fb7b0f03909"
uuid = "b8865327-cd53-5732-bb35-84acbb429228"
version = "3.4.0"

[[deps.Unzip]]
git-tree-sha1 = "ca0969166a028236229f63514992fc073799bb78"
uuid = "41fe7b60-77ed-43a1-b4f0-825fd5a5650d"
version = "0.2.0"

[[deps.Wayland_jll]]
deps = ["Artifacts", "Expat_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "ed8d92d9774b077c53e1da50fd81a36af3744c1c"
uuid = "a2964d1f-97da-50d4-b82a-358c7fce9d89"
version = "1.21.0+0"

[[deps.Wayland_protocols_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4528479aa01ee1b3b4cd0e6faef0e04cf16466da"
uuid = "2381bf8a-dfd0-557d-9999-79630e7b1b91"
version = "1.25.0+0"

[[deps.WeakRefStrings]]
deps = ["DataAPI", "InlineStrings", "Parsers"]
git-tree-sha1 = "b1be2855ed9ed8eac54e5caff2afcdb442d52c23"
uuid = "ea10d353-3f73-51f8-a26c-33c1cb351aa5"
version = "1.4.2"

[[deps.WebIO]]
deps = ["AssetRegistry", "Base64", "Distributed", "FunctionalCollections", "JSON", "Logging", "Observables", "Pkg", "Random", "Requires", "Sockets", "UUIDs", "WebSockets", "Widgets"]
git-tree-sha1 = "976d0738247f155d0dcd77607edea644f069e1e9"
uuid = "0f1e0344-ec1d-5b48-a673-e5cf874b6c29"
version = "0.8.20"

[[deps.WebSockets]]
deps = ["Base64", "Dates", "HTTP", "Logging", "Sockets"]
git-tree-sha1 = "f91a602e25fe6b89afc93cf02a4ae18ee9384ce3"
uuid = "104b5d7c-a370-577a-8038-80a2059c5097"
version = "1.5.9"

[[deps.Widgets]]
deps = ["Colors", "Dates", "Observables", "OrderedCollections"]
git-tree-sha1 = "fcdae142c1cfc7d89de2d11e08721d0f2f86c98a"
uuid = "cc8bc4a8-27d6-5769-a93b-9d913e69aa62"
version = "0.6.6"

[[deps.WoodburyMatrices]]
deps = ["LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "de67fa59e33ad156a590055375a30b23c40299d3"
uuid = "efce3f68-66dc-5838-9240-27a6d6f5f9b6"
version = "0.5.5"

[[deps.WorkerUtilities]]
git-tree-sha1 = "cd1659ba0d57b71a464a29e64dbc67cfe83d54e7"
uuid = "76eceee3-57b5-4d4a-8e66-0e911cebbf60"
version = "1.6.1"

[[deps.XML2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "93c41695bc1c08c46c5899f4fe06d6ead504bb73"
uuid = "02c8fc9c-b97f-50b9-bbe4-9be30ff0a78a"
version = "2.10.3+0"

[[deps.XSLT_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgcrypt_jll", "Libgpg_error_jll", "Libiconv_jll", "Pkg", "XML2_jll", "Zlib_jll"]
git-tree-sha1 = "91844873c4085240b95e795f692c4cec4d805f8a"
uuid = "aed1982a-8fda-507f-9586-7b0439959a61"
version = "1.1.34+0"

[[deps.Xorg_libX11_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxcb_jll", "Xorg_xtrans_jll"]
git-tree-sha1 = "5be649d550f3f4b95308bf0183b82e2582876527"
uuid = "4f6342f7-b3d2-589e-9d20-edeb45f2b2bc"
version = "1.6.9+4"

[[deps.Xorg_libXau_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4e490d5c960c314f33885790ed410ff3a94ce67e"
uuid = "0c0b7dd1-d40b-584c-a123-a41640f87eec"
version = "1.0.9+4"

[[deps.Xorg_libXcursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXfixes_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "12e0eb3bc634fa2080c1c37fccf56f7c22989afd"
uuid = "935fb764-8cf2-53bf-bb30-45bb1f8bf724"
version = "1.2.0+4"

[[deps.Xorg_libXdmcp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fe47bd2247248125c428978740e18a681372dd4"
uuid = "a3789734-cfe1-5b06-b2d0-1dd0d9d62d05"
version = "1.1.3+4"

[[deps.Xorg_libXext_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "b7c0aa8c376b31e4852b360222848637f481f8c3"
uuid = "1082639a-0dae-5f34-9b06-72781eeb8cb3"
version = "1.3.4+4"

[[deps.Xorg_libXfixes_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "0e0dc7431e7a0587559f9294aeec269471c991a4"
uuid = "d091e8ba-531a-589c-9de9-94069b037ed8"
version = "5.0.3+4"

[[deps.Xorg_libXi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll", "Xorg_libXfixes_jll"]
git-tree-sha1 = "89b52bc2160aadc84d707093930ef0bffa641246"
uuid = "a51aa0fd-4e3c-5386-b890-e753decda492"
version = "1.7.10+4"

[[deps.Xorg_libXinerama_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll"]
git-tree-sha1 = "26be8b1c342929259317d8b9f7b53bf2bb73b123"
uuid = "d1454406-59df-5ea1-beac-c340f2130bc3"
version = "1.1.4+4"

[[deps.Xorg_libXrandr_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "34cea83cb726fb58f325887bf0612c6b3fb17631"
uuid = "ec84b674-ba8e-5d96-8ba1-2a689ba10484"
version = "1.5.2+4"

[[deps.Xorg_libXrender_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "19560f30fd49f4d4efbe7002a1037f8c43d43b96"
uuid = "ea2f1a96-1ddc-540d-b46f-429655e07cfa"
version = "0.9.10+4"

[[deps.Xorg_libpthread_stubs_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "6783737e45d3c59a4a4c4091f5f88cdcf0908cbb"
uuid = "14d82f49-176c-5ed1-bb49-ad3f5cbd8c74"
version = "0.1.0+3"

[[deps.Xorg_libxcb_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "XSLT_jll", "Xorg_libXau_jll", "Xorg_libXdmcp_jll", "Xorg_libpthread_stubs_jll"]
git-tree-sha1 = "daf17f441228e7a3833846cd048892861cff16d6"
uuid = "c7cfdc94-dc32-55de-ac96-5a1b8d977c5b"
version = "1.13.0+3"

[[deps.Xorg_libxkbfile_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "926af861744212db0eb001d9e40b5d16292080b2"
uuid = "cc61e674-0454-545c-8b26-ed2c68acab7a"
version = "1.1.0+4"

[[deps.Xorg_xcb_util_image_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "0fab0a40349ba1cba2c1da699243396ff8e94b97"
uuid = "12413925-8142-5f55-bb0e-6d7ca50bb09b"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxcb_jll"]
git-tree-sha1 = "e7fd7b2881fa2eaa72717420894d3938177862d1"
uuid = "2def613f-5ad1-5310-b15b-b15d46f528f5"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_keysyms_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "d1151e2c45a544f32441a567d1690e701ec89b00"
uuid = "975044d2-76e6-5fbe-bf08-97ce7c6574c7"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_renderutil_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "dfd7a8f38d4613b6a575253b3174dd991ca6183e"
uuid = "0d47668e-0667-5a69-a72c-f761630bfb7e"
version = "0.3.9+1"

[[deps.Xorg_xcb_util_wm_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "e78d10aab01a4a154142c5006ed44fd9e8e31b67"
uuid = "c22f9ab0-d5fe-5066-847c-f4bb1cd4e361"
version = "0.4.1+1"

[[deps.Xorg_xkbcomp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxkbfile_jll"]
git-tree-sha1 = "4bcbf660f6c2e714f87e960a171b119d06ee163b"
uuid = "35661453-b289-5fab-8a00-3d9160c6a3a4"
version = "1.4.2+4"

[[deps.Xorg_xkeyboard_config_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xkbcomp_jll"]
git-tree-sha1 = "5c8424f8a67c3f2209646d4425f3d415fee5931d"
uuid = "33bec58e-1273-512f-9401-5d533626f822"
version = "2.27.0+4"

[[deps.Xorg_xtrans_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "79c31e7844f6ecf779705fbc12146eb190b7d845"
uuid = "c5fb5394-a638-5e4d-96e5-b29de1b5cf10"
version = "1.4.0+3"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.2.12+3"

[[deps.Zstd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "c6edfe154ad7b313c01aceca188c05c835c67360"
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.4+0"

[[deps.fzf_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "868e669ccb12ba16eaf50cb2957ee2ff61261c56"
uuid = "214eeab7-80f7-51ab-84ad-2988db7cef09"
version = "0.29.0+0"

[[deps.libaom_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "3a2ea60308f0996d26f1e5354e10c24e9ef905d4"
uuid = "a4ae2306-e953-59d6-aa16-d00cac43593b"
version = "3.4.0+0"

[[deps.libass_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "5982a94fcba20f02f42ace44b9894ee2b140fe47"
uuid = "0ac62f75-1d6f-5e53-bd7c-93b484bb37c0"
version = "0.15.1+0"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl", "OpenBLAS_jll"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.1.1+0"

[[deps.libfdk_aac_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "daacc84a041563f965be61859a36e17c4e4fcd55"
uuid = "f638f0a6-7fb0-5443-88ba-1cc74229b280"
version = "2.0.2+0"

[[deps.libpng_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "94d180a6d2b5e55e447e2d27a29ed04fe79eb30c"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.38+0"

[[deps.libvorbis_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Ogg_jll", "Pkg"]
git-tree-sha1 = "b910cb81ef3fe6e78bf6acee440bda86fd6ae00c"
uuid = "f27f6e37-5d2b-51aa-960f-b287f2bc3b7a"
version = "1.3.7+1"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.48.0+0"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.4.0+0"

[[deps.x264_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fea590b89e6ec504593146bf8b988b2c00922b2"
uuid = "1270edf5-f2f9-52d2-97e9-ab00b5d0237a"
version = "2021.5.5+0"

[[deps.x265_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "ee567a171cce03570d77ad3a43e90218e38937a9"
uuid = "dfaa095f-4041-5dcd-9319-2fabd8486b76"
version = "3.5.0+0"

[[deps.xkbcommon_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Wayland_jll", "Wayland_protocols_jll", "Xorg_libxcb_jll", "Xorg_xkeyboard_config_jll"]
git-tree-sha1 = "9ebfc140cc56e8c2156a15ceac2f0302e327ac0a"
uuid = "d8fb68d0-12a3-5cfd-a85a-d49703b185fd"
version = "1.4.1+0"
"""

# ╔═╡ Cell order:
# ╠═dff3c71e-2ed4-4663-8441-e10b1b2f76d6
# ╟─8fb0a34b-80b0-4862-ae94-0a777e430ff2
# ╟─9786047f-c383-4269-af6f-90039f3caf74
# ╟─958e8517-f147-4a61-a3c1-3335f8e8c36d
# ╠═02ef7e5c-0858-4da4-af5e-1dcc6060c6ef
# ╟─fbd49b80-c4e9-4b8d-83d7-ac3c3c958a38
# ╟─db362343-b3be-4a2d-b4b0-002417560b3a
# ╠═48a07b04-52e6-4dbe-beb5-5b45e148136e
# ╠═604677b8-ec66-4137-ab47-0121874457bb
# ╟─00540701-8536-4603-a337-05ca3df89ad7
# ╟─5837be60-44d6-4219-b0b3-940de66337de
# ╠═b5645969-e265-46d5-b676-1404b1b9d69c
# ╟─6519e8ad-ea5c-428a-90be-a81ef87195c0
# ╟─9ee89705-ad5a-465a-b704-2dab939c3ed7
# ╠═a587a40c-582b-48fc-a3bf-aed21c9c5875
# ╟─72484660-f9ea-4be1-bc76-f15102cc2515
# ╠═d5537edd-28ec-46b0-b9d9-f347b772ccb2
# ╠═b4206d72-52ab-44ec-8336-027762356df7
# ╟─b9e43bd4-6d88-4b08-8a3a-fabef87a0ece
# ╠═53fb9803-ae4d-47ad-b1cd-d354d119afda
# ╟─e761d48f-6b04-4c8d-be81-18edc017e95e
# ╟─e1bcc629-1c9c-4d8e-90f5-ada601350fef
# ╠═4600aa8f-b941-4939-97a2-ff8da9d27e8d
# ╠═904d7e4d-9ca1-4d36-b7fd-cfe0793d77e1
# ╠═c3005d9d-06c7-4cf1-8f02-7bed71f9b03c
# ╠═e0ae4760-94eb-45ae-90d1-a6ece8d49df3
# ╟─0c6bf5ba-00cd-47bc-bae8-4b93b2732cc5
# ╟─fd3b7a31-5655-4a98-b1f5-e4a5deca2f4e
# ╠═ce6e9347-5974-4bb4-aba8-bfea8b17d751
# ╟─beae22b7-3fbb-429f-85e3-d7fd4ac82d5f
# ╠═f19de1fb-6512-479f-88a6-7e9cf84a3849
# ╠═1d56f85e-2df2-4ee0-8e91-4b435f8baa76
# ╠═9e27cd53-d648-4a06-b47e-39ec06a1ae02
# ╠═eeb04698-1cb7-4c26-a1db-966fd57142f4
# ╟─2c9f360e-fc82-4f10-b192-a1785465add5
# ╠═124c13a2-5682-49c3-a7e3-b20f405d27a7
# ╟─b3d9f580-5b85-40f7-8864-0cda14b4a8d4
# ╟─eb46ddaf-4cb2-4cb6-a187-49e3d27e48f0
# ╟─afbac097-eac1-4b9b-a9d3-30783f84be24
# ╟─333c9e3b-6f1d-4993-9368-190c8527a86d
# ╠═c4165e26-7fd6-4833-9c09-e28b7be27b01
# ╠═c590c041-dd90-4f3d-895f-ff0607a0d0ee
# ╟─72268cf2-9a1c-40e3-a5e7-e3ac7851dc70
# ╠═e95f78d2-9960-4c6b-837d-4c1af186bcb0
# ╟─d7820f71-2a92-4693-9dc4-fcfd5ecb05a5
# ╟─94bb684a-81d8-4255-887b-4f00edee3d0d
# ╠═f78da47b-15db-474b-86b7-163378bc6e95
# ╠═4e6cf955-5d1b-4eff-8243-e3826650ca6b
# ╠═de6c5c60-e2d5-4eef-87c0-facd98c447cd
# ╟─8bd3dabc-3a37-477c-ae71-dbf34bd16c4b
# ╠═cbf222b6-3cac-40b6-9d75-fe0ba7a0058a
# ╟─49a33464-8e04-4238-ba7c-d16dc1acf101
# ╠═6c1290ec-cccb-4a99-b2d9-9843cb1a56d2
# ╟─9c33577c-cb72-47c8-b3f9-11a9ee2d0536
# ╠═4984ee85-31fe-43f1-a606-d597cb9d58b2
# ╟─715d2091-a709-4559-abfd-0bbe86a70781
# ╟─2febf6b3-477f-44e1-a204-9f6a892fa1cd
# ╟─9458e108-925a-45f5-8669-5ad735859b4e
# ╠═b6f9b431-82ff-44d0-b374-7e6a8ad401e2
# ╠═e3311988-cc8c-476d-8f6e-3a37ed1d48df
# ╠═b479eb87-dcbe-4171-a03a-9153832d2139
# ╠═f1a4ba68-9a48-49a9-8760-3521cab29896
# ╟─c888d313-634b-4d5c-bc31-184f3efaa884
# ╟─6df86af7-2804-4961-94da-cffed0410125
# ╟─e4f5b9d5-06b2-4296-9694-ef97ef0b2c5f
# ╟─7886a080-ef8f-4bf0-9f5c-045e1636eece
# ╟─7e05295f-3e1e-4800-a989-ae81f7b59de0
# ╟─b0da2855-91a4-463a-bbda-ab8d8160b850
# ╠═57149050-83ad-42c4-807a-616deec1ae8c
# ╟─01eb7746-2ab2-45b3-a602-904f930e1b23
# ╠═74ac32f4-e44b-4dea-b239-dc2ee39c734f
# ╟─b305634d-e636-4fd1-b235-c9174b9c7b52
# ╟─8004fe42-7406-4df9-bd27-1f9a7b4c8b07
# ╠═f4551c02-98a1-4485-abc5-2991483ecdaa
# ╠═909791fe-b893-4597-aeb9-499019ebeca7
# ╠═b375353a-0bd1-4bd0-a5f7-fe60ddad626b
# ╟─49873af1-4979-4962-a69a-9006af80bfa5
# ╠═e9f4f135-588e-4679-96f2-0fb8a9bdf016
# ╠═79b23ab0-dbe1-4290-b095-36772bb09363
# ╠═3a27f686-8612-473c-a005-3c452df03237
# ╠═abdfcc54-1652-4169-9b0a-af2257d2ef8e
# ╟─a1317f7f-a939-4421-b31f-775f1e943123
# ╟─160b93db-1455-4319-99aa-db19780638aa
# ╠═d1cbb5da-44db-40dd-b1b8-745d0ee77f82
# ╠═2cdb6e50-19ea-4fc1-9bc8-80a5fb30a528
# ╠═3a847626-61a7-4d0e-87c1-e8676a6406f5
# ╠═8c70a0bf-4346-4fd0-8d77-61008be3773b
# ╠═1c53a451-b1dd-41b7-80cd-cb8da81737f1
# ╟─806708b8-5e43-4591-9eeb-38db217c6a01
# ╟─a07b9a55-a745-4c54-9393-d4b8752475a2
# ╟─43a09104-5948-49ce-9a7b-7b938ffb26d3
# ╟─52eb82fb-97b6-42be-a9bb-63a1f8624c45
# ╠═91ee8380-7d05-47c1-a88d-80e9b09195e3
# ╠═c3ba53fe-ccef-4dd1-88f4-72365c880603
# ╠═a7711e7b-288d-46c0-910d-51dc6cbc22e4
# ╠═c036ff8b-267d-4673-8582-9b39ac4502ff
# ╠═d2ebc549-3879-40c6-ad91-ccef317358ab
# ╠═8701c933-08fb-48f1-bb62-cdcf94c4ba4f
# ╟─6004e00c-3a31-429b-bfe7-4bce0b8a15bb
# ╟─14822bc5-0827-4c8b-91db-6d057fae8e33
# ╟─fdef7b51-3285-4388-ba09-91a4a6f22bd7
# ╟─aaa21bda-3d18-4785-8166-f73be94a85eb
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
