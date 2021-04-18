Authors: nsu-hackspace-team
   
Project structure:
 
    wave_detector directory:
	
	Hydra provides instrumentation for image-based froth analysis
    based on features dynamics. Detected features are related 
    to froth movement. Features flow characterized by median velocity
    and direction. Dispersion of direction provides ability to 
    detect waves on froth surface.

	Morph provides instrumentation for image-based froth analysis
    based on contour structure. Detected countours are related 
    to froth glares. Countours system characterized by speed 
    of change (related to steadiness of froth flow) and integral
    parameter of detected bubbles mass center.

	run by command:
	python3 hydra.py videofile

    collected_data:
	dataset of different type processes, collected by our instruments 
    from videos

    neural_segmentation directory:
	Not very fast, but robust froth size classifier
