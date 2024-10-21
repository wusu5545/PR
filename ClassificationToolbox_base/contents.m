% Classification GUI and toolbox
% Version 1.0
%
% GUI start commands
%
%		classifier			- Start the classification GUI
%		enter_distributions - Starts the parameter input screen (used by classifier)
%		multialgorithms		- Start the algorithm comparison screen
%       show_algorithms     - Show possible algorithms for classification, clustering and feature selection
%
% Preprocessing methods
%
%
% Parametric classification algorithms
%
%		NearestNeighbor		- NearestNeighbor Classification Algorithm
%
% Non-parametric classification algorithms
%
%
% Feature selection
%
%
% Error estimation
%
%		calculate_error		        - Calculates the classification error given a decision surface
%		classification_error        - Used by claculate_error
%		classify_paramteric         - Builds a decision region for multi-Gaussian distributions
%
% Error bounds
%
%		Bhattacharyya 
%		Chernoff
%		Discriminability
%
% GUI housekeeping functions
%
%		calculate_region	        - Finds the data scatter region
%		classifier_commands	        - Classifier screen commands
%		click_points				- Graphically enter a distribution
%		enter_distribution_commands	- Used by enter_distributions
%       feature_selection           - The feature selection GUI open when data with more than 2D is loaded
%       feature_selection_commands  - The commands file for the feature selection GUI
%       FindParameters              - A GUI for finding the optimal parameters for a classifier
%       FindParametersFunctions     - The commands file for FindParameters
%       GaussianParameters          - Opens a GUI for displaying the gaussian parameters of a distribution
%		generate_data_set	        - Generate a data set given Gaussian parameters
%		high_histogram              - Generate a histogram for high-dimensional data
%		load_file			        - Load data files
%		make_a_draw			        - Randomly find indices from a data set
%       multialgorithms_commands    - Multialgorithms screen comands
%       plot_process                - Plot partition centers during the algorithm execution
%		plot_scatter		        - Make a scatter plot of a data set
%       Predict_performance         - Predict performance of algorithms from their learning curves
%       process_params              - Read a parameter vector and return it's components
%		read_algorithms	            - Reads an algorithm file into a data structure
%		start_classify		        - Main function used by classifier
%		voronoi_regions		        - Plot Voronoi regions
%
% Data sets (Ending _data means that the file contains patterns, 
%                   _params means that the file contains the distribution parameters)
%
%
%____________________________________________________________________________________
%  Elad Yom-Tov (elad@ieee.org) and David Stork
%  Technion - Israel Institute of Technology
%  Haifa, Israel