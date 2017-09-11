"use strict";

document.addEventListener('DOMContentLoaded', function() {
	var options = {
	  name: 'dagre',
      nodeSep: 3, // the separation between adjacent nodes in the same rank
	  edgeSep: 5, // the separation between adjacent edges in the same rank
	  minLen: function( edge ){ return 1; }, // number of ranks to keep between the source and target of the edge
	  edgeWeight: function( edge ){ return 2; }, // higher weight edges are generally made shorter and straighter than lower weight edges
	  fit: true, // whether to fit to viewport
	  spacingFactor: 1.1, // Applies a multiplicative factor (>0) to expand or compress the overall area that the nodes take up
	  nodeDimensionsIncludeLabels: true // whether labels should be included in determining the space used by a node (default true)
	};

	var nodeInfo = getGraphNodesAndEdges();
	nodeInfo.then(function(nodesArray) {
		var cy = window.cy = cytoscape({
			container: document.getElementById('cy'),
		    elements: nodesArray,
		    layout: options,
		    style: [
			{
			    selector: "node",
			    style: {
			        shape: 'roundrectangle',
			        label: 'data(name)',
                    'font-size' : 30,
			        'border-width': 3,
                    'border-style': 'outset',
			        width: 'label',
			        'color': '#000000',
			        'text-valign': 'center',
					'background-image': 'icons/node.png',
			        padding: 10,
		    	}
		    },
			{
			    selector: "node.parent",
			    style: {
			        shape: 'roundrectangle',
			        label: 'data(name)',
                    'font-size' : 30,
			        'border-width': 10,
                    'border-color': 'black',
			        width: 'label',
			        'color': 'black',
			        'text-valign': 'center',
			        padding: 10,
					'compound-sizing-wrt-labels': 'include',
                    'background-image' : 'icons/parent.png',
                    'text-rotation' : '90deg',
                    'text-margin-x' : 10
		    	}
		    },
            {
			    selector: "node.parent > node",
			    style: {
			        opacity : 0
		    	}
		    },
            {
			    selector: "node.arrayFeatureExtractor",
			    style: {
			        shape: 'roundrectangle',
			        width: 'label',
			        label: 'data(name)',
			        'border-width': 2,
			        'color': '#000000',
					padding: 10,
			        'font-weight': 'bold',
					'background-image': 'icons/arrayFeatureExtractor.png'
		    	}
		    },
            {
			    selector: "node.categoricalMapping",
			    style: {
			        shape: 'roundrectangle',
			        width: 'label',
			        label: 'data(name)',
			        'border-width': 2,
			        'color': '#000000',
					padding: 10,
			        'font-weight': 'bold',
					'background-image': 'icons/categoricalMapping.png'
		    	}
		    },
            {
			    selector: "node.dictVectorizer",
			    style: {
			        shape: 'roundrectangle',
			        width: 'label',
			        label: 'data(name)',
			        'border-width': 2,
			        'color': '#000000',
					padding: 10,
			        'font-weight': 'bold',
					'background-image': 'icons/dictVectorizer.png'
		    	}
		    },
            {
			    selector: "node.featureVectorizer",
			    style: {
			        shape: 'roundrectangle',
			        width: 'label',
			        label: 'data(name)',
			        'border-width': 2,
			        'color': '#000000',
					padding: 10,
			        'font-weight': 'bold',
					'background-image': 'icons/featureVectorizer.png'
		    	}
		    },
            {
			    selector: "node.glmClassifier",
			    style: {
			        shape: 'roundrectangle',
			        width: 'label',
			        label: 'data(name)',
			        'border-width': 2,
			        'color': '#000000',
					padding: 10,
			        'font-weight': 'bold',
					'background-image': 'icons/glmClassifier.png'
		    	}
		    },
            {
			    selector: "node.glmRegressor",
			    style: {
			        shape: 'roundrectangle',
			        width: 'label',
			        label: 'data(name)',
			        'border-width': 2,
			        'color': '#000000',
					padding: 10,
			        'font-weight': 'bold',
					'background-image': 'icons/glmRegressor.png'
		    	}
		    },
            {
			    selector: "node.identity",
			    style: {
			        shape: 'roundrectangle',
			        width: 'label',
			        label: 'data(name)',
			        'border-width': 2,
			        'color': '#000000',
					padding: 10,
			        'font-weight': 'bold',
					'background-image': 'icons/identity.png'
		    	}
		    },
            {
			    selector: "node.imputer",
			    style: {
			        shape: 'roundrectangle',
			        width: 'label',
			        label: 'data(name)',
			        'border-width': 2,
			        'color': '#000000',
					padding: 10,
			        'font-weight': 'bold',
					'background-image': 'icons/imputer.png'
		    	}
		    },
            {
			    selector: "node.neuralNetwork",
			    style: {
			        shape: 'roundrectangle',
			        width: 'label',
			        label: 'data(name)',
			        'border-width': 2,
			        'color': '#000000',
					padding: 10,
			        'font-weight': 'bold',
					'background-image': 'icons/neuralNetwork.png'
		    	}
		    },
            {
			    selector: "node.neuralNetworkClassifier",
			    style: {
			        shape: 'roundrectangle',
			        width: 'label',
			        label: 'data(name)',
			        'border-width': 2,
			        'color': '#000000',
					padding: 10,
			        'font-weight': 'bold',
					'background-image': 'icons/neuralNetworkClassifier.png'
		    	}
		    },
            {
			    selector: "node.neuralNetworkRegressor",
			    style: {
			        shape: 'roundrectangle',
			        width: 'label',
			        label: 'data(name)',
			        'border-width': 2,
			        'color': '#000000',
					padding: 10,
			        'font-weight': 'bold',
					'background-image': 'icons/neuralNetworkRegressor.png'
		    	}
		    },
            {
			    selector: "node.normalizer",
			    style: {
			        shape: 'roundrectangle',
			        width: 'label',
			        label: 'data(name)',
			        'border-width': 2,
			        'color': '#000000',
					padding: 10,
			        'font-weight': 'bold',
					'background-image': 'icons/normalizer.png'
		    	}
		    },
            {
			    selector: "node.oneHotEncoder",
			    style: {
			        shape: 'roundrectangle',
			        width: 'label',
			        label: 'data(name)',
			        'border-width': 2,
			        'color': '#000000',
					padding: 10,
			        'font-weight': 'bold',
					'background-image': 'icons/oneHotEncoder.png'
		    	}
		    },
            {
			    selector: "node.scaler",
			    style: {
			        shape: 'roundrectangle',
			        width: 'label',
			        label: 'data(name)',
			        'border-width': 2,
			        'color': '#000000',
					padding: 10,
			        'font-weight': 'bold',
					'background-image': 'icons/scaler.png'
		    	}
		    },
            {
			    selector: "node.supportVectorClassifier",
			    style: {
			        shape: 'roundrectangle',
			        width: 'label',
			        label: 'data(name)',
			        'border-width': 2,
			        'color': '#000000',
					padding: 10,
			        'font-weight': 'bold',
					'background-image': 'icons/supportVectorClassifier.png'
		    	}
		    },
            {
			    selector: "node.supportVectorRegressor",
			    style: {
			        shape: 'roundrectangle',
			        width: 'label',
			        label: 'data(name)',
			        'border-width': 2,
			        'color': '#000000',
					padding: 10,
			        'font-weight': 'bold',
					'background-image': 'icons/supportVectorRegressor.png'
		    	}
		    },
            {
			    selector: "node.treeEnsembleClassifier",
			    style: {
			        shape: 'roundrectangle',
			        width: 'label',
			        label: 'data(name)',
			        'border-width': 2,
			        'color': '#000000',
					padding: 10,
			        'font-weight': 'bold',
					'background-image': 'icons/treeEnsembleClassifier.png'
		    	}
		    },
            {
			    selector: "node.treeEnsembleRegressor",
			    style: {
			        shape: 'roundrectangle',
			        width: 'label',
			        label: 'data(name)',
			        'border-width': 2,
			        'color': '#000000',
					padding: 10,
			        'font-weight': 'bold',
					'background-image': 'icons/treeEnsembleRegressor.png'
		    	}
		    },
            {
			    selector: "node.convolution",
			    style: {
			        shape: 'roundrectangle',
                    'font-weight': 'bold',
			        label: 'data(name)',
                    'font-size' : 30,
			        'border-width': 2,
			        'color': '#FFFFFF',
			        width: 'label',
			        'text-valign': 'center',
					'background-image': 'icons/convolution.png'
		    	}
		    },
		    {
			    selector: "node.pooling",
			    style: {
			        shape: 'roundrectangle',
			        width: 'label',
			        label: 'data(name)',
			        'border-width': 2,
			        'color': '#FFFFFF',
			        'font-weight': 'bold',
					'background-image': 'icons/pooling.png'
		    	}
		    },
		    {
			    selector: "node.activation",
			    style: {
			        shape: 'roundrectangle',
			        width: 'label',
			        label: 'data(name)',
			        'border-width': 2,
			        'color': '#000000',
					padding: 10,
			        'font-weight': 'bold',
					'background-image': 'icons/activation.png'
		    	}
		    },
            {
			    selector: "node.add",
			    style: {
			        shape: 'roundrectangle',
			        width: 'label',
			        label: 'data(name)',
			        'border-width': 2,
			        'color': '#000000',
					padding: 10,
			        'font-weight': 'bold',
					'background-image': 'icons/add.png'
		    	}
		    },
            {
			    selector: "node.average",
			    style: {
			        shape: 'roundrectangle',
			        width: 'label',
			        label: 'data(name)',
			        'border-width': 2,
			        'color': '#000000',
					padding: 10,
			        'font-weight': 'bold',
					'background-image': 'icons/average.png'
		    	}
		    },
            {
			    selector: "node.batchnorm",
			    style: {
			        shape: 'roundrectangle',
			        width: 'label',
			        label: 'data(name)',
			        'border-width': 2,
			        'color': '#000000',
					padding: 10,
			        'font-weight': 'bold',
					'background-image': 'icons/batchnorm.png'
		    	}
		    },
            {
			    selector: "node.biDirectionalLSTM",
			    style: {
			        shape: 'roundrectangle',
			        width: 'label',
			        label: 'data(name)',
			        'border-width': 2,
			        'color': '#000000',
					padding: 10,
			        'font-weight': 'bold',
					'background-image': 'icons/biDirectionalLSTM.png'
		    	}
		    },
            {
			    selector: "node.bias",
			    style: {
			        shape: 'roundrectangle',
			        width: 'label',
			        label: 'data(name)',
			        'border-width': 2,
			        'color': '#000000',
					padding: 10,
			        'font-weight': 'bold',
					'background-image': 'icons/bias.png'
		    	}
		    },
            {
			    selector: "node.concat",
			    style: {
			        shape: 'roundrectangle',
			        width: 'label',
			        label: 'data(name)',
			        'border-width': 2,
			        'color': '#000000',
					padding: 10,
			        'font-weight': 'bold',
					'background-image': 'icons/concat.png'
		    	}
		    },
            {
			    selector: "node.crop",
			    style: {
			        shape: 'roundrectangle',
			        width: 'label',
			        label: 'data(name)',
			        'border-width': 2,
			        'color': '#000000',
					padding: 10,
			        'font-weight': 'bold',
					'background-image': 'icons/crop.png'
		    	}
		    },
            {
			    selector: "node.dot",
			    style: {
			        shape: 'roundrectangle',
			        width: 'label',
			        label: 'data(name)',
			        'border-width': 2,
			        'color': '#000000',
					padding: 10,
			        'font-weight': 'bold',
					'background-image': 'icons/dot.png'
		    	}
		    },
            {
			    selector: "node.embedding",
			    style: {
			        shape: 'roundrectangle',
			        width: 'label',
			        label: 'data(name)',
			        'border-width': 2,
			        'color': '#000000',
					padding: 10,
			        'font-weight': 'bold',
					'background-image': 'icons/embedding.png'
		    	}
		    },
            {
			    selector: "node.flatten",
			    style: {
			        shape: 'roundrectangle',
			        width: 'label',
			        label: 'data(name)',
			        'border-width': 2,
			        'color': '#000000',
					padding: 10,
			        'font-weight': 'bold',
					'background-image': 'icons/flatten.png'
		    	}
		    },
            {
			    selector: "node.gru",
			    style: {
			        shape: 'roundrectangle',
			        width: 'label',
			        label: 'data(name)',
			        'border-width': 2,
			        'color': '#000000',
					padding: 10,
			        'font-weight': 'bold',
					'background-image': 'icons/gru.png'
		    	}
		    },
            {
			    selector: "node.innerProduct",
			    style: {
			        shape: 'roundrectangle',
			        width: 'label',
			        label: 'data(name)',
			        'border-width': 2,
			        'color': '#000000',
					padding: 10,
			        'font-weight': 'bold',
					'background-image': 'icons/innerProduct.png'
		    	}
		    },
            {
			    selector: "node.input",
			    style: {
			        shape: 'roundrectangle',
			        width: 'label',
			        label: 'data(name)',
			        'border-width': 2,
			        'color': '#000000',
					padding: 10,
			        'font-weight': 'bold',
					'background-image': 'icons/input.png'
		    	}
		    },
            {
			    selector: "node.output",
			    style: {
			        shape: 'roundrectangle',
			        width: 'label',
			        label: 'data(name)',
			        'border-width': 2,
			        'color': '#000000',
					padding: 10,
			        'font-weight': 'bold',
					'background-image': 'icons/output.png'
		    	}
		    },
            {
			    selector: "node.l2normalize",
			    style: {
			        shape: 'roundrectangle',
			        width: 'label',
			        label: 'data(name)',
			        'border-width': 2,
			        'color': '#000000',
					padding: 10,
			        'font-weight': 'bold',
					'background-image': 'icons/l2normalize.png'
		    	}
		    },
            {
			    selector: "node.loadConstant",
			    style: {
			        shape: 'roundrectangle',
			        width: 'label',
			        label: 'data(name)',
			        'border-width': 2,
			        'color': '#000000',
					padding: 10,
			        'font-weight': 'bold',
					'background-image': 'icons/loadConstant.png'
		    	}
		    },
            {
			    selector: "node.lrn",
			    style: {
			        shape: 'roundrectangle',
			        width: 'label',
			        label: 'data(name)',
			        'border-width': 2,
			        'color': '#000000',
					padding: 10,
			        'font-weight': 'bold',
					'background-image': 'icons/lrn.png'
		    	}
		    },
            {
			    selector: "node.max",
			    style: {
			        shape: 'roundrectangle',
			        width: 'label',
			        label: 'data(name)',
			        'border-width': 2,
			        'color': '#000000',
					padding: 10,
			        'font-weight': 'bold',
					'background-image': 'icons/max.png'
		    	}
		    },
            {
			    selector: "node.min",
			    style: {
			        shape: 'roundrectangle',
			        width: 'label',
			        label: 'data(name)',
			        'border-width': 2,
			        'color': '#000000',
					padding: 10,
			        'font-weight': 'bold',
					'background-image': 'icons/min.png'
		    	}
		    },
            {
			    selector: "node.multiply",
			    style: {
			        shape: 'roundrectangle',
			        width: 'label',
			        label: 'data(name)',
			        'border-width': 2,
			        'color': '#000000',
					padding: 10,
			        'font-weight': 'bold',
					'background-image': 'icons/multiply.png'
		    	}
		    },
            {
			    selector: "node.mvn",
			    style: {
			        shape: 'roundrectangle',
			        width: 'label',
			        label: 'data(name)',
			        'border-width': 2,
			        'color': '#000000',
					padding: 10,
			        'font-weight': 'bold',
					'background-image': 'icons/mvn.png'
		    	}
		    },
            {
			    selector: "node.padding",
			    style: {
			        shape: 'roundrectangle',
			        width: 'label',
			        label: 'data(name)',
			        'border-width': 2,
			        'color': '#000000',
					padding: 10,
			        'font-weight': 'bold',
					'background-image': 'icons/padding.png'
		    	}
		    },
            {
			    selector: "node.permute",
			    style: {
			        shape: 'roundrectangle',
			        width: 'label',
			        label: 'data(name)',
			        'border-width': 2,
			        'color': '#000000',
					padding: 10,
			        'font-weight': 'bold',
					'background-image': 'icons/permute.png'
		    	}
		    },
            {
			    selector: "node.pooling",
			    style: {
			        shape: 'roundrectangle',
			        width: 'label',
			        label: 'data(name)',
			        'border-width': 2,
			        'color': '#FFFFFF',
					padding: 10,
			        'font-weight': 'bold',
					'background-image': 'icons/pooling.png'
		    	}
		    },
            {
			    selector: "node.reduce",
			    style: {
			        shape: 'roundrectangle',
			        width: 'label',
			        label: 'data(name)',
			        'border-width': 2,
			        'color': '#000000',
					padding: 10,
			        'font-weight': 'bold',
					'background-image': 'icons/reduce.png'
		    	}
		    },
            {
			    selector: "node.reorganizeData",
			    style: {
			        shape: 'roundrectangle',
			        width: 'label',
			        label: 'data(name)',
			        'border-width': 2,
			        'color': '#000000',
					padding: 10,
			        'font-weight': 'bold',
					'background-image': 'icons/reorganizeData.png'
		    	}
		    },
            {
			    selector: "node.reshape",
			    style: {
			        shape: 'roundrectangle',
			        width: 'label',
			        label: 'data(name)',
			        'border-width': 2,
			        'color': '#000000',
					padding: 10,
			        'font-weight': 'bold',
					'background-image': 'icons/reshape.png'
		    	}
		    },
            {
			    selector: "node.scale",
			    style: {
			        shape: 'roundrectangle',
			        width: 'label',
			        label: 'data(name)',
			        'border-width': 2,
			        'color': '#000000',
					padding: 10,
			        'font-weight': 'bold',
					'background-image': 'icons/scale.png'
		    	}
		    },
            {
			    selector: "node.sequenceRepeat",
			    style: {
			        shape: 'roundrectangle',
			        width: 'label',
			        label: 'data(name)',
			        'border-width': 2,
			        'color': '#000000',
					padding: 10,
			        'font-weight': 'bold',
					'background-image': 'icons/sequenceRepeat.png'
		    	}
		    },
            {
			    selector: "node.simpleRecurrent",
			    style: {
			        shape: 'roundrectangle',
			        width: 'label',
			        label: 'data(name)',
			        'border-width': 2,
			        'color': '#000000',
					padding: 10,
			        'font-weight': 'bold',
					'background-image': 'icons/simpleRecurrent.png'
		    	}
		    },
            {
			    selector: "node.slice",
			    style: {
			        shape: 'roundrectangle',
			        width: 'label',
			        label: 'data(name)',
			        'border-width': 2,
			        'color': '#000000',
					padding: 10,
			        'font-weight': 'bold',
					'background-image': 'icons/slice.png'
		    	}
		    },
            {
			    selector: "node.softmax",
			    style: {
			        shape: 'roundrectangle',
			        width: 'label',
			        label: 'data(name)',
			        'border-width': 2,
			        'color': '#000000',
					padding: 10,
			        'font-weight': 'bold',
					'background-image': 'icons/softmax.png'
		    	}
		    },
            {
			    selector: "node.split",
			    style: {
			        shape: 'roundrectangle',
			        width: 'label',
			        label: 'data(name)',
			        'border-width': 2,
			        'color': '#000000',
					padding: 10,
			        'font-weight': 'bold',
					'background-image': 'icons/split.png'
		    	}
		    },
            {
			    selector: "node.unary",
			    style: {
			        shape: 'roundrectangle',
			        width: 'label',
			        label: 'data(name)',
			        'border-width': 2,
			        'color': '#000000',
					padding: 10,
			        'font-weight': 'bold',
					'background-image': 'icons/unary.png'
		    	}
		    },
            {
			    selector: "node.uniDirectionalLSTM",
			    style: {
			        shape: 'roundrectangle',
			        width: 'label',
			        label: 'data(name)',
			        'border-width': 2,
			        'color': '#000000',
					padding: 10,
			        'font-weight': 'bold',
					'background-image': 'icons/uniDirectionalLSTM.png'
		    	}
		    },
            {
			    selector: "node.upsample",
			    style: {
			        shape: 'roundrectangle',
			        width: 'label',
			        label: 'data(name)',
			        'border-width': 2,
			        'color': '#000000',
					padding: 10,
			        'font-weight': 'bold',
					'background-image': 'icons/upsample.png'
		    	}
		    },
		    {
		    	selector: "edge",
		    	style: {
		    		'curve-style': 'bezier',
		    		'control-point-weights': 1,
		    		'line-color': '#111111',
					'color' : '#000000',
					'border-width': 5,
		    		'target-arrow-shape': 'triangle',
		    		'target-arrow-color': '#111111',
					 label: 'data(label)',
					'text-background-opacity': 0,
					'text-background-color': '#ffffff',
					'text-background-shape': 'rectangle',
					'text-border-style': 'solid',
					'text-border-opacity': 0,
					'text-border-width': '1px',
					'text-border-color': 'darkgray',
					'text-opacity': 0
		    	}
		    }
		    ]
		});
        cy.fit();
        var childNodeCollection = cy.elements("node.parent > node");
        var childEdges  = childNodeCollection.connectedEdges();
        childEdges.style({'opacity': 0});

		cy.$('node').on('mouseover', function(e){
            var ele = e.target;
            // console.log(ele.classes());

		    var keys = Object.keys(ele.data('info'));
		    var div = document.getElementById('node-info');
		    var content = '<br />';
		    content += '<div class="subtitle">Parameters</div>';
		    content += '<br />';
		    for (var i = keys.length - 1; i >= 0; i--) {
		  	    if (keys[i] != 'desc') {
                    var val = ele.data('info')[keys[i]];
                    content += keys[i].toUpperCase() + ' : ' + val.charAt(0).toUpperCase() + val.slice(1) + '<br />';
                }
            }
            if (ele.data('info')["desc"] != undefined) {
                content += '<br /><br /><div class="subtitle">Description</div><br />';
                content += ele.data('info')["desc"] + '<br />';
            }
            div.innerHTML = content;
		});

		cy.$('node').on('mouseout', function(e){
			var div = document.getElementById('node-info');
			div.innerHTML = '';
		});

		cy.on('click', 'node.parent', function(evt){
		    var node = evt.target;
		    console.log( 'tapped ' + node.id() );
            node.children().style({'opacity': 1});
            node.style({'color' : '#d5e1df'});
            var selectedChildNodeCollection = node.children();
            var selectedChildEdges  = selectedChildNodeCollection.connectedEdges();
            console.log(selectedChildNodeCollection);
            selectedChildEdges.style({'opacity' : 1});
            node.connectedEdges().style({'opacity' : 0});
            cy.animate({
                fit : {
                    eles : selectedChildNodeCollection,
                    padding : 20
                }
            }, {
                duration: 500
            });

		});

		$('#label-switch').on('click', function(e) {
			if (cy.edges().style().textOpacity == 0) {
				cy.edges().style({
					'text-opacity': 1,
					'text-background-opacity': 1,
					'text-border-opacity': 1
				});
			}
			else {
				cy.edges().style({
					'text-opacity': 0,
					'text-background-opacity': 0,
					'text-border-opacity': 0
				});
			}
		});

		$('#reset-state').on('click', function (e) {
			var childNodes = cy.$("node.parent > node");
			childNodes.style({
					"opacity": 0
				});
			childNodes.connectedEdges().style({
				'opacity': 0
			});
			var parentNodes = cy.$("node.parent");
			parentNodes.style({
				'color': 'black'
			});
			parentNodes.connectedEdges().style({
				'opacity': 1
			});
			cy.fit();
		});
        
	});

});


function getGraphNodesAndEdges() {
	var graphPromise  = $.ajax({
		url: 'model.json',
		type: 'GET',
		dataType: 'json',
		contentType: "application/json; charset=utf-8",
	})
	.then(function(msg) {
		return msg;
	}
	);
	return Promise.resolve(graphPromise);

}