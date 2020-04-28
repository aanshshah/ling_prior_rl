var relation_mapping = {'bucket': {'RelatedTo': ['bucket', 'abstraction', 'artifact', 'instrumentality', 'object', 'entity', 'containerful', 'container', 'whole']}, 'soil': {'RelatedTo': ['soil', 'dirt', 'clay']}, 'abstraction': {'RelatedTo': ['bucket', 'abstraction', 'dirt', 'shovel', 'beef']}, 'pig': {'RelatedTo': ['pig', 'unpleasant_person', 'meat', 'artifact', 'instrumentality', 'object', 'container', 'even-toed_ungulate', 'physical_entity', 'person']}, 'dirt': {'RelatedTo': ['soil', 'abstraction', 'dirt', 'matter', 'material', 'speech_act', 'physical_entity', 'body_waste', 'event']}, 'unpleasant_person': {'RelatedTo': ['pig', 'unpleasant_person', 'cow']}, 'log': {'RelatedTo': ['log', 'matter', 'instrument', 'artifact', 'material', 'substance', 'instrumentality', 'physical_entity', 'device']}, 'tool': {'RelatedTo': ['tool', 'pickaxe', 'shears', 'shovel', 'instrument', 'object', 'hoe', 'axe'], 'IsA': ['object']}, 'pickaxe': {'RelatedTo': ['tool', 'pickaxe', 'artifact', 'instrumentality', 'object', 'edge_tool', 'physical_entity', 'whole']}, 'shears': {'RelatedTo': ['tool', 'shears', 'artifact', 'instrumentality', 'object', 'entity', 'edge_tool', 'event', 'whole']}, 'wool': {'RelatedTo': ['tool', 'shears', 'wool', 'matter', 'artifact', 'material', 'sheep', 'object', 'physical_entity', 'whole'], 'AtLocation': ['sheep']}, 'solid': {'RelatedTo': ['solid', 'beef', 'stone', 'porkchop']}, 'shovel': {'RelatedTo': ['abstraction', 'tool', 'shovel', 'instrument', 'artifact', 'instrumentality', 'object', 'entity', 'containerful', 'device', 'whole']}, 'meat': {'RelatedTo': ['pig', 'meat', 'beef', 'cow', 'porkchop']}, 'matter': {'RelatedTo': ['dirt', 'log', 'solid', 'matter', 'beef', 'material', 'substance', 'water', 'porkchop', 'clay']}, 'instrument': {'RelatedTo': ['log', 'tool', 'instrument', 'sword', 'device'], 'UsedFor': ['tool']}, 'cobblestone': {'RelatedTo': ['cobblestone', 'artifact', 'stone', 'object', 'physical_entity', 'whole']}, 'beef': {'RelatedTo': ['abstraction', 'solid', 'meat', 'matter', 'beef', 'speech_act', 'cow', 'bovid', 'even-toed_ungulate', 'cattle', 'physical_entity', 'event']}, 'artifact': {'RelatedTo': ['bucket', 'pig', 'log', 'pickaxe', 'shears', 'wool', 'shovel', 'cobblestone', 'artifact', 'substance', 'stone', 'hoe', 'water', 'sword', 'person', 'event', 'axe'], 'IsA': ['artifact', 'substance', 'object', 'event'], 'PartOf': ['artifact'], 'UsedFor': ['artifact', 'substance', 'object', 'person', 'event'], 'AtLocation': ['artifact', 'object']}, 'material': {'RelatedTo': ['dirt', 'log', 'wool', 'material', 'substance', 'stone', 'clay']}, 'water_bucket': {'RelatedTo': ['water_bucket']}, 'sheep': {'RelatedTo': ['pig', 'wool', 'meat', 'sheep', 'object', 'entity', 'cow', 'bovid', 'even-toed_ungulate', 'physical_entity', 'person', 'whole']}, 'substance': {'RelatedTo': ['log', 'substance', 'stone', 'water', 'clay'], 'IsA': ['artifact', 'substance', 'object', 'event'], 'PartOf': ['artifact', 'substance'], 'AtLocation': ['container']}, 'instrumentality': {'RelatedTo': ['bucket', 'pig', 'log', 'pickaxe', 'shears', 'shovel', 'instrumentality', 'hoe', 'sword', 'axe']}, 'stone': {'RelatedTo': ['dirt', 'solid', 'matter', 'cobblestone', 'artifact', 'material', 'substance', 'stone', 'object', 'water', 'natural_object', 'person']}, 'object': {'RelatedTo': ['bucket', 'pig', 'pickaxe', 'shears', 'wool', 'shovel', 'cobblestone', 'sheep', 'stone', 'object', 'hoe', 'cow', 'clay', 'farmland', 'sword', 'axe'], 'IsA': ['artifact', 'substance', 'object'], 'PartOf': ['artifact', 'substance', 'object', 'event']}, 'entity': {'RelatedTo': ['bucket', 'shears', 'shovel', 'sheep', 'entity', 'cow', 'porkchop']}, 'hoe': {'RelatedTo': ['tool', 'artifact', 'instrumentality', 'object', 'hoe', 'physical_entity', 'whole']}, 'containerful': {'RelatedTo': ['bucket', 'shovel', 'containerful']}, 'container': {'RelatedTo': ['bucket', 'pig', 'container']}, 'speech_act': {'RelatedTo': ['dirt', 'beef', 'speech_act']}, 'cow': {'RelatedTo': ['unpleasant_person', 'beef', 'object', 'entity', 'cow', 'bovid', 'cattle', 'physical_entity', 'person', 'whole']}, 'edge_tool': {'RelatedTo': ['pickaxe', 'shears', 'edge_tool', 'axe']}, 'bovid': {'RelatedTo': ['beef', 'sheep', 'cow', 'bovid']}, 'even-toed_ungulate': {'RelatedTo': ['pig', 'beef', 'sheep', 'even-toed_ungulate']}, 'cattle': {'RelatedTo': ['beef', 'cow', 'cattle']}, 'water': {'AtLocation': ['bucket', 'container'], 'RelatedTo': ['matter', 'artifact', 'substance', 'water', 'physical_entity', 'body_waste', 'device']}, 'porkchop': {'RelatedTo': ['solid', 'meat', 'matter', 'entity', 'porkchop', 'physical_entity']}, 'physical_entity': {'RelatedTo': ['pig', 'dirt', 'log', 'pickaxe', 'wool', 'cobblestone', 'beef', 'sheep', 'hoe', 'cow', 'water', 'porkchop', 'physical_entity', 'farmland', 'sword', 'axe']}, 'body_waste': {'RelatedTo': ['dirt', 'water', 'body_waste']}, 'natural_object': {'RelatedTo': ['stone', 'natural_object', 'clay']}, 'clay': {'RelatedTo': ['soil', 'matter', 'material', 'substance', 'object', 'natural_object', 'clay', 'person', 'whole']}, 'farmland': {'RelatedTo': ['object', 'physical_entity', 'farmland']}, 'sword': {'RelatedTo': ['instrument', 'artifact', 'instrumentality', 'object', 'physical_entity', 'sword', 'device', 'whole'], 'AtLocation': ['stone']}, 'person': {'RelatedTo': ['pig', 'sheep', 'stone', 'object', 'cow', 'clay', 'person'], 'PartOf': ['object', 'person'], 'IsA': ['person']}, 'event': {'RelatedTo': ['dirt', 'shears', 'beef', 'event'], 'IsA': ['event'], 'PartOf': ['event']}, 'device': {'RelatedTo': ['log', 'shovel', 'sword', 'device']}, 'whole': {'RelatedTo': ['bucket', 'pickaxe', 'shears', 'wool', 'shovel', 'cobblestone', 'sheep', 'hoe', 'cow', 'clay', 'sword', 'whole', 'axe']}, 'axe': {'RelatedTo': ['tool', 'artifact', 'instrumentality', 'object', 'edge_tool', 'physical_entity', 'device', 'whole', 'axe']}};

var all_words = [];
for (var key of Object.keys(relation_mapping)) {
    all_words.push(key);
  }
var select = document.getElementById("myDropdown");
for(var i = 0; i < all_words.length; i++) {
    var opt = all_words[i];
    var el = document.createElement("a");
    el.href = "javascript:createGraph(\'"+opt+"\')";
    el.textContent = opt;
    el.value = opt;
    select.appendChild(el);
}

		/* When the user clicks on the button,
toggle between hiding and showing the dropdown content */
function myFunction() {
  document.getElementById("myDropdown").classList.toggle("show");
}

function filterFunction() {
  var input, filter, ul, li, a, i;
  input = document.getElementById("myInput");
  filter = input.value.toUpperCase();
  div = document.getElementById("myDropdown");
  a = div.getElementsByTagName("a");
  for (i = 0; i < a.length; i++) {
    txtValue = a[i].textContent || a[i].innerText;
    if (txtValue.toUpperCase().indexOf(filter) > -1) {
      a[i].style.display = "";
    } else {
      a[i].style.display = "none";
    }
  }
}

function clearSVG(){
	d3.select("svg").selectAll('*').remove();
}


function buildConnections(relations, mapping, start_word){
  var link = []; //list of {target: index, source: index, strength: 1, group: relations_index} for each relation
  var word_to_idx = {};
  word_to_idx[start_word] = 0;
  var source_idx = 0;
  var curr_idx = 1;
  var source_word = start_word;
  var stack = new Array();
  stack.push(source_word);

  while(stack.length > 0){
  	source_word = stack.pop();

  	source_idx = word_to_idx[source_word];
  	for(var i=0; i<relations.length; i++){
	    var relation_name = relations[i];
	    
	    var target_words = mapping[source_word][relation_name];
	    if (target_words){
	    	for(var j=0; j<target_words.length; j++){
	    		var target_word = target_words[j];
	    		if(target_word in word_to_idx){
	    			var target_word_idx = word_to_idx[target_word];
	    		}else{
	    			var target_word_idx = curr_idx;
	    			word_to_idx[target_word] = target_word_idx;
	    			curr_idx++;
	    			stack.push(target_word);
	    		}

	    		link.push({target: target_word_idx, source: source_idx, strength: 1, group: i});
	    	}
	    }
	}	
  }
  var nodes = [];
  for (var key of Object.keys(word_to_idx)) {
    nodes.push({id: word_to_idx[key], label: key});
  }
  return [nodes, link];
}

function createGraph(start_word){
	myFunction();
	clearSVG();
	var relations = ["RelatedTo", "UsedFor", "AtLocation", "IsA", "PartOf"];
	var toVisualize = buildConnections(relations, relation_mapping, start_word);
	var baseNodes = toVisualize[0];
	var baseLinks = toVisualize[1];
	var width = 2000 //window.innerWidth
	var height = 2000//window.innerHeight

		function getNeighbors(node) {
		  return baseLinks.reduce(function (neighbors, link) {
		      if (link.target.id === node.id) {
		        neighbors.push(link.source.id)
		      } else if (link.source.id === node.id) {
		        neighbors.push(link.target.id)
		      }
		      return neighbors
		    },
		    [node.id]
		  )
		}

		function isNeighborLink(node, link) {
		  return link.target.id === node.id || link.source.id === node.id
		}

		function getConnectNodeColor(node, neighbors){
			neighbors.indexOf(node.id) ? 'white' : 'grey'
		}
		function getTextColor(node, neighbors) {
		  return neighbors.indexOf(node.id) ? 'green' : 'black'
		}
		function getLinkColor2(node, link) {
		  return isNeighborLink(node, link) ? 'green' : '#E5E5E5'
		}


var nodes = [...baseNodes]
var links = [...baseLinks]

function getNeighbors(node) {
  return baseLinks.reduce(function (neighbors, link) {
      if (link.target.id === node.id) {
        neighbors.push(link.source.id)
      } else if (link.source.id === node.id) {
        neighbors.push(link.target.id)
      }
      return neighbors
    },
    [node.id]
  )
}


function getLinkColor(link) {
	var color = 'grey';
	switch(link.group){
		case 0:
			color = 'red';
			break;
		case 1:
			color = 'blue';
			break;
		case 2:
			color = 'green';
			break;
		case 3:
			color = 'purple';
			break;
		case 4:
			color = 'grey';
			break;
		default:
			color = 'grey';
	}
	return color;
}

var svg = d3.select('svg')
svg.attr('width', width).attr('height', height)

var linkElements,
  nodeElements,
  textElements

// we use svg groups to logically group the elements together
var linkGroup = svg.append('g').attr('class', 'links')
var nodeGroup = svg.append('g').attr('class', 'nodes')
var textGroup = svg.append('g').attr('class', 'texts')

// we use this reference to select/deselect
// after clicking the same element twice
var selectedId

// simulation setup with all forces
var linkForce = d3
  .forceLink()
  .id(function (link) { return link.id })
  .strength(function (link) { return link.strength })

var simulation = d3
  .forceSimulation()
  .force('link', linkForce)
  .force('charge', d3.forceManyBody().strength(-1200))
  .force('center', d3.forceCenter(width / 3, height / 3.5))

var dragDrop = d3.drag().on('start', function (node) {
  node.fx = node.x
  node.fy = node.y
}).on('drag', function (node) {
  simulation.alphaTarget(0.7).restart()
  node.fx = d3.event.x
  node.fy = d3.event.y
}).on('end', function (node) {
  if (!d3.event.active) {
    simulation.alphaTarget(0)
  }
  node.fx = null
  node.fy = null
})

// select node is called on every click
// we either update the data according to the selection
// or reset the data if the same node is clicked twice
function selectNode(selectedNode) {
  if (selectedId === selectedNode.id) {
    selectedId = undefined
    resetData()
    updateSimulation()
  } else {
    selectedId = selectedNode.id
    updateData(selectedNode)
    updateSimulation()
  }

  var neighbors = getNeighbors(selectedNode)

  // we modify the styles to highlight selected nodes
  nodeElements.attr('fill', function (node) { return 'grey' })
  textElements.attr('fill', function (node) { return getTextColor(node, neighbors) })
  linkElements.attr('stroke', function (link) { return getLinkColor(link) })
}

// this helper simple adds all nodes and links
// that are missing, to recreate the initial state
function resetData() {
  var nodeIds = nodes.map(function (node) { return node.id })

  baseNodes.forEach(function (node) {
    if (nodeIds.indexOf(node.id) === -1) {
      nodes.push(node)
    }
  })

  links = baseLinks
}

// diffing and mutating the data
function updateData(selectedNode) {
  var neighbors = getNeighbors(selectedNode)
  var newNodes = baseNodes.filter(function (node) {
    return neighbors.indexOf(node.id) > -1 || node.level === 1
  })

  var diff = {
    removed: nodes.filter(function (node) { return newNodes.indexOf(node) === -1 }),
    added: newNodes.filter(function (node) { return nodes.indexOf(node) === -1 })
  }

  diff.removed.forEach(function (node) { nodes.splice(nodes.indexOf(node), 1) })
  diff.added.forEach(function (node) { nodes.push(node) })

  links = baseLinks.filter(function (link) {
    return link.target.id === selectedNode.id || link.source.id === selectedNode.id
  })
}

function updateGraph() {
  // links
  linkElements = linkGroup.selectAll('line')
    .data(links, function (link) {
      return link.target.id + link.source.id
    })

  linkElements.exit().remove()

  var linkEnter = linkElements
    .enter().append('line')
    .attr('stroke-width', 1)
    .attr('stroke', function (link) { return getLinkColor(link) })

  linkElements = linkEnter.merge(linkElements)

  // nodes
  nodeElements = nodeGroup.selectAll('circle')
    .data(nodes, function (node) { return node.id })

  nodeElements.exit().remove()

  var nodeEnter = nodeElements
    .enter()
    .append('circle')
    .attr('r', 10)
    .attr('fill', function (node) { return node.level === 1 ? 'red' : 'gray' })
    .call(dragDrop)
    // we link the selectNode method here
    // to update the graph on every click
    .on('click', selectNode)

  nodeElements = nodeEnter.merge(nodeElements)

  // texts
  textElements = textGroup.selectAll('text')
    .data(nodes, function (node) { return node.id })

  textElements.exit().remove()

  var textEnter = textElements
    .enter()
    .append('text')
    .text(function (node) { return node.label })
    .attr('font-size', 15)
    .attr('dx', 15)
    .attr('dy', 4)

  textElements = textEnter.merge(textElements)
}

function updateSimulation() {
  updateGraph()

  simulation.nodes(nodes).on('tick', () => {
    nodeElements
      .attr('cx', function (node) { return node.x })
      .attr('cy', function (node) { return node.y })
    textElements
      .attr('x', function (node) { return node.x })
      .attr('y', function (node) { return node.y })
    linkElements
      .attr('x1', function (link) { return link.source.x })
      .attr('y1', function (link) { return link.source.y })
      .attr('x2', function (link) { return link.target.x })
      .attr('y2', function (link) { return link.target.y })
  })

  simulation.force('link').links(links)
  simulation.alphaTarget(0.7).restart()
}

// last but not least, we call updateSimulation
// to trigger the initial render
updateSimulation()
}