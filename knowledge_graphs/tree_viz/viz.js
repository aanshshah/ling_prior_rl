var relation_mapping = {"laptop": {"RelatedTo": ["laptop", "whole", "device", "physical_entity", "instrumentality", "artifact"]}, "television": {"RelatedTo": ["television", "medium", "whole", "equipment", "physical_entity", "instrumentality", "electronic_equipment", "artifact"], "IsA": ["device", "act", "person"], "PartOf": ["television"], "AtLocation": ["cabinet"]}, "countertop": {"RelatedTo": ["countertop", "whole", "physical_entity", "surface", "artifact"]}, "room": {"RelatedTo": ["room", "area", "box", "container", "compartment", "toilet", "bed", "furniture", "cabinet"], "UsedFor": ["furniture"]}, "matter": {"RelatedTo": ["matter", "lettuce", "substance", "sandwich", "newspaper", "material", "omelette", "fried_egg", "potato", "bread", "solid", "tomato", "dirt", "pencil", "apple", "toilet_paper", "egg"]}, "creation": {"RelatedTo": ["creation", "newspaper", "painting", "statue", "book"]}, "area": {"RelatedTo": ["area", "box", "toilet", "cabinet", "surface", "pen", "measure"]}, "tableware": {"RelatedTo": ["tableware", "bowl", "fork", "cup", "plate", "spoon"]}, "fridge": {"RelatedTo": ["fridge", "whole", "physical_entity", "artifact", "home_appliance"]}, "paper": {"RelatedTo": ["paper", "substance", "fabric", "medium", "newspaper", "writing_implement", "material", "implement", "tool", "object", "instrument", "painting", "book", "product", "surface", "pen", "pencil", "toilet_paper"], "IsA": ["material", "solid"], "AtLocation": ["cabinet"]}, "box": {"RelatedTo": ["room", "area", "paper", "box", "container", "seat", "structure", "organism", "material", "shape", "whole", "vascular_plant", "object", "physical_entity", "containerful", "compartment", "abstraction", "condition", "instrumentality", "furniture", "woody_plant", "artifact", "measure"], "AtLocation": ["pen"]}, "kitchen_appliance": {"RelatedTo": ["kitchen_appliance", "microwave", "toaster"]}, "creditcard": {"RelatedTo": ["creditcard"]}, "container": {"RelatedTo": ["box", "container", "pot", "garbage_can", "bowl", "mug", "whole", "cup", "pan", "plate", "physical_entity", "watering_can", "instrumentality", "bathtub", "artifact", "spoon"], "AtLocation": ["food", "cabinet"]}, "shower_door": {"RelatedTo": ["shower_door"]}, "spray_bottle": {"RelatedTo": ["spray_bottle"]}, "food": {"RelatedTo": ["food", "lettuce", "substance", "sandwich", "omelette", "fried_egg", "herb", "cup", "plate", "potato", "bread", "tomato", "apple", "egg"], "IsA": ["food", "substance", "object", "act", "artifact"], "PartOf": ["food"], "AtLocation": ["fridge", "container", "bowl", "cabinet", "spoon"]}, "nutriment": {"RelatedTo": ["nutriment", "sandwich", "omelette", "fried_egg", "plate"]}, "scrub_brush": {"RelatedTo": ["scrub_brush", "implement", "whole", "physical_entity", "instrumentality", "artifact"]}, "protective_covering": {"RelatedTo": ["protective_covering", "plate", "cabinet", "blinds"]}, "egg_shell": {"RelatedTo": ["egg_shell"]}, "pot": {"RelatedTo": ["container", "pot", "substance", "bowl", "whole", "causal_agent", "containerful", "abstraction", "instrumentality", "artifact", "cooking_utensil", "plumbing_fixture", "measure", "vessel"], "IsA": ["tool"]}, "garbage_can": {"RelatedTo": ["container", "garbage_can", "whole", "physical_entity", "instrumentality", "artifact"]}, "sports_equipment": {"RelatedTo": ["sports_equipment", "whole", "equipment", "plate", "physical_entity", "instrumentality", "artifact"]}, "lettuce": {"RelatedTo": ["matter", "food", "lettuce", "organism", "living_thing", "whole", "herb", "vascular_plant", "produce", "abstraction", "measure"]}, "group": {"RelatedTo": ["group", "book", "blinds"], "IsA": ["food", "group", "act", "artifact", "art"], "PartOf": ["group", "object", "artifact"]}, "foodstuff": {"RelatedTo": ["foodstuff", "potato", "bread", "egg"]}, "substance": {"RelatedTo": ["pot", "substance", "sandwich", "omelette", "fried_egg", "potato", "dirt", "pencil", "toilet_paper", "egg"], "IsA": ["food", "group", "substance", "object", "artifact", "attribute"], "PartOf": ["substance", "artifact"], "AtLocation": ["container"]}, "vacuum_cleaner": {"RelatedTo": ["vacuum_cleaner", "whole", "physical_entity", "artifact", "home_appliance"]}, "mirror": {"RelatedTo": ["mirror", "whole", "object", "device", "physical_entity", "instrumentality", "artifact"], "IsA": ["surface"]}, "fabric": {"RelatedTo": ["fabric", "material", "towel", "cloth"]}, "bowl": {"RelatedTo": ["tableware", "container", "food", "bowl", "structure", "dish", "shape", "whole", "concave_shape", "activity", "equipment", "plate", "physical_entity", "containerful", "abstraction", "solid", "act", "instrumentality", "artifact", "measure", "vessel"], "AtLocation": ["cabinet", "sink"]}, "microwave": {"RelatedTo": ["kitchen_appliance", "microwave", "whole", "physical_entity", "process", "artifact", "home_appliance"]}, "lightswitch": {"RelatedTo": ["lightswitch"]}, "seat": {"RelatedTo": ["area", "box", "seat", "object", "device", "toilet", "chair", "furniture", "surface"]}, "sandwich": {"RelatedTo": ["matter", "food", "nutriment", "substance", "sandwich", "dish", "physical_entity", "entity"], "AtLocation": ["fridge", "plate"]}, "medium": {"RelatedTo": ["television", "medium", "newspaper"]}, "barsoap": {"RelatedTo": ["barsoap"]}, "lamp": {"RelatedTo": ["room", "lamp", "whole", "device", "physical_entity", "instrumentality", "furniture", "artifact", "candle"]}, "newspaper": {"RelatedTo": ["matter", "creation", "paper", "medium", "newspaper", "material", "whole", "instrumentality", "product", "artifact"]}, "cellphone": {"RelatedTo": ["cellphone", "whole", "equipment", "physical_entity", "instrumentality", "electronic_equipment", "artifact"]}, "keychain": {"RelatedTo": ["keychain"]}, "structure": {"RelatedTo": ["creation", "box", "bowl", "structure", "shape", "object", "support", "toilet", "cabinet", "knob", "pen"]}, "writing_implement": {"RelatedTo": ["writing_implement", "pen", "pencil"]}, "dish": {"RelatedTo": ["sandwich", "dish", "omelette", "fried_egg", "plate"], "IsA": ["plate"], "UsedFor": ["food"], "AtLocation": ["cabinet", "sink"]}, "organism": {"RelatedTo": ["box", "lettuce", "organism", "toaster", "mug", "potato", "chair", "watch", "tomato", "plunger", "hanger"], "IsA": ["process"]}, "material": {"RelatedTo": ["substance", "fabric", "newspaper", "material", "dirt", "cloth", "toilet_paper"]}, "shape": {"RelatedTo": ["bowl", "shape", "cup", "object", "statue", "knob", "pencil", "art", "knife", "attribute"], "IsA": ["shape", "artifact", "attribute"], "PartOf": ["shape", "artifact", "attribute"]}, "implement": {"RelatedTo": ["scrub_brush", "implement", "fork", "plunger", "pen", "pencil", "knife"]}, "toaster": {"RelatedTo": ["kitchen_appliance", "organism", "toaster", "living_thing", "whole", "tool", "object", "causal_agent", "device", "physical_entity", "bread", "entity", "person", "home_appliance"]}, "mug": {"RelatedTo": ["container", "organism", "mug", "whole", "cup", "physical_entity", "containerful", "abstraction", "instrumentality", "body_part", "artifact", "person", "measure", "vessel"], "IsA": ["dish"], "AtLocation": ["cabinet"]}, "living_thing": {"RelatedTo": ["lettuce", "toaster", "living_thing", "chair", "tomato", "plunger", "egg", "hanger"], "IsA": ["attribute"]}, "omelette": {"RelatedTo": ["matter", "food", "nutriment", "substance", "dish", "omelette", "physical_entity", "entity"]}, "alarm_clock": {"RelatedTo": ["alarm_clock", "whole", "device", "physical_entity", "instrument", "instrumentality", "artifact", "timepiece"]}, "whole": {"RelatedTo": ["laptop", "television", "countertop", "fridge", "box", "container", "scrub_brush", "pot", "garbage_can", "sports_equipment", "lettuce", "vacuum_cleaner", "mirror", "bowl", "microwave", "lamp", "newspaper", "cellphone", "toaster", "mug", "alarm_clock", "whole", "fork", "cup", "pan", "plate", "watering_can", "potato", "towel", "toilet", "painting", "bed", "chair", "statue", "remote_control", "book", "cabinet", "tomato", "knob", "bathtub", "plunger", "pen", "cloth", "pencil", "spoon", "apple", "sink", "blinds", "egg", "pillow", "candle", "knife", "hanger"]}, "concave_shape": {"RelatedTo": ["bowl", "concave_shape", "cup"]}, "convex_shape": {"RelatedTo": ["convex_shape", "knob", "knife"]}, "state": {"RelatedTo": ["area", "state", "toilet", "dirt"], "IsA": ["substance", "shape", "state", "process", "act", "attribute"], "PartOf": ["state"]}, "activity": {"RelatedTo": ["activity", "toilet", "painting", "chair", "watch"]}, "fried_egg": {"RelatedTo": ["matter", "food", "nutriment", "substance", "dish", "fried_egg", "physical_entity", "entity"], "AtLocation": ["plate"]}, "tool": {"RelatedTo": ["box", "implement", "tool", "fork", "object", "instrument", "plunger", "knife"], "IsA": ["object"]}, "fork": {"RelatedTo": ["tableware", "implement", "whole", "tool", "fork", "object", "physical_entity", "cutlery", "abstraction", "instrumentality", "artifact", "spoon", "knife", "attribute", "figure"], "AtLocation": ["plate"]}, "herb": {"RelatedTo": ["lettuce", "herb", "tomato"]}, "cup": {"RelatedTo": ["tableware", "container", "food", "shape", "whole", "concave_shape", "cup", "containerful", "abstraction", "plant_organ", "solid", "instrumentality", "cabinet", "artifact", "measure", "vessel", "attribute"], "IsA": ["vessel"], "AtLocation": ["sink"]}, "vascular_plant": {"RelatedTo": ["box", "lettuce", "vascular_plant", "potato", "tomato"]}, "object": {"RelatedTo": ["toaster", "fork", "object", "apple"], "IsA": ["group", "substance", "shape", "object", "artifact"], "PartOf": ["food", "substance", "object", "artifact"]}, "causal_agent": {"RelatedTo": ["pot", "toaster", "causal_agent"]}, "pan": {"RelatedTo": ["container", "pot", "dish", "implement", "whole", "pan", "physical_entity", "instrument", "instrumentality", "artifact", "cooking_utensil", "vessel"], "AtLocation": ["cabinet"]}, "device": {"RelatedTo": ["laptop", "mirror", "lamp", "alarm_clock", "device", "plate", "chair", "remote_control", "watch", "plunger", "candle", "knife", "hanger"]}, "equipment": {"RelatedTo": ["television", "sports_equipment", "bowl", "cellphone", "tool", "equipment", "plate"]}, "support": {"RelatedTo": ["group", "structure", "support", "plate", "chair", "hanger"]}, "plate": {"RelatedTo": ["room", "tableware", "paper", "container", "food", "nutriment", "protective_covering", "sports_equipment", "bowl", "dish", "shape", "implement", "whole", "tool", "fork", "cup", "object", "device", "equipment", "support", "plate", "physical_entity", "instrument", "containerful", "cutlery", "cabinet", "surface", "artifact", "spoon", "measure", "vessel", "knife"], "IsA": ["dish"], "AtLocation": ["cabinet", "sink"]}, "physical_entity": {"RelatedTo": ["laptop", "television", "countertop", "fridge", "box", "container", "scrub_brush", "garbage_can", "sports_equipment", "vacuum_cleaner", "mirror", "bowl", "microwave", "sandwich", "lamp", "cellphone", "toaster", "mug", "omelette", "alarm_clock", "fried_egg", "fork", "pan", "plate", "physical_entity", "watering_can", "potato", "bread", "towel", "toilet", "painting", "bed", "chair", "statue", "remote_control", "cabinet", "tomato", "dirt", "knob", "bathtub", "plunger", "pen", "cloth", "spoon", "apple", "sink", "toilet_paper", "egg", "pillow", "candle", "knife", "hanger"]}, "watering_can": {"RelatedTo": ["container", "whole", "physical_entity", "watering_can", "instrumentality", "artifact"]}, "starches": {"RelatedTo": ["starches", "potato", "bread"]}, "potato": {"RelatedTo": ["matter", "food", "foodstuff", "substance", "organism", "whole", "vascular_plant", "physical_entity", "starches", "potato", "solanaceous_vegetable", "produce"], "AtLocation": ["pot", "microwave", "pan", "plate", "dirt", "sink"]}, "coffee_machine": {"RelatedTo": ["coffee_machine"]}, "instrument": {"RelatedTo": ["alarm_clock", "tool", "device", "instrument", "chair", "watch", "knife"], "UsedFor": ["tool"]}, "containerful": {"RelatedTo": ["box", "pot", "bowl", "mug", "cup", "plate", "containerful", "spoon"]}, "solanaceous_vegetable": {"RelatedTo": ["potato", "solanaceous_vegetable", "tomato"]}, "bread": {"RelatedTo": ["matter", "box", "food", "foodstuff", "sandwich", "toaster", "whole", "pan", "physical_entity", "starches", "bread", "abstraction", "solid", "product", "measure"], "PartOf": ["sandwich"], "AtLocation": ["sandwich", "toaster"]}, "towel": {"RelatedTo": ["fabric", "whole", "physical_entity", "towel", "artifact"]}, "produce": {"RelatedTo": ["food", "lettuce", "potato", "produce", "tomato", "apple"]}, "cutlery": {"RelatedTo": ["fork", "cutlery", "spoon"]}, "compartment": {"RelatedTo": ["box", "compartment", "cabinet"]}, "abstraction": {"RelatedTo": ["box", "pot", "lettuce", "bowl", "mug", "fork", "cup", "bread", "abstraction", "statue", "watch", "book", "dirt", "pencil", "spoon", "toilet_paper", "blinds", "knife"]}, "toilet": {"RelatedTo": ["room", "area", "seat", "structure", "whole", "state", "activity", "physical_entity", "toilet", "furniture", "artifact", "plumbing_fixture"]}, "plant_organ": {"RelatedTo": ["cup", "plant_organ", "apple"]}, "house_plant": {"RelatedTo": ["house_plant"]}, "process": {"RelatedTo": ["microwave", "process", "sink"], "IsA": ["organism", "process", "act"], "PartOf": ["organism", "state", "process"]}, "painting": {"RelatedTo": ["creation", "whole", "activity", "physical_entity", "painting", "occupation", "act", "artifact", "art"]}, "bed": {"RelatedTo": ["room", "area", "box", "container", "structure", "alarm_clock", "whole", "tool", "object", "device", "equipment", "support", "physical_entity", "instrument", "bed", "furniture", "surface", "artifact", "person", "pillow", "natural_depression"], "AtLocation": ["room"]}, "solid": {"RelatedTo": ["bowl", "cup", "solid", "knob", "knife"]}, "occupation": {"RelatedTo": ["painting", "occupation", "chair"]}, "act": {"RelatedTo": ["group", "bowl", "painting", "act", "chair", "dirt"], "IsA": ["group", "organism", "state", "process", "act", "artifact", "attribute"], "PartOf": ["group", "state", "process", "act", "artifact"]}, "condition": {"RelatedTo": ["box", "dish", "shape", "state", "condition", "dirt", "product"]}, "instrumentality": {"RelatedTo": ["laptop", "television", "box", "container", "scrub_brush", "pot", "garbage_can", "sports_equipment", "mirror", "bowl", "lamp", "newspaper", "cellphone", "mug", "alarm_clock", "fork", "cup", "pan", "watering_can", "instrumentality", "chair", "remote_control", "watch", "cabinet", "bathtub", "plunger", "pen", "pencil", "spoon", "candle", "knife", "hanger"]}, "chair": {"RelatedTo": ["room", "seat", "organism", "material", "living_thing", "whole", "activity", "tool", "object", "device", "support", "physical_entity", "instrument", "occupation", "act", "instrumentality", "chair", "furniture", "surface", "artifact", "person"], "AtLocation": ["room"]}, "statue": {"RelatedTo": ["creation", "structure", "shape", "whole", "object", "physical_entity", "abstraction", "painting", "solid", "statue", "artifact", "person", "art", "attribute", "figure"]}, "remote_control": {"RelatedTo": ["whole", "device", "physical_entity", "instrumentality", "remote_control", "artifact"], "AtLocation": ["television"]}, "watch": {"RelatedTo": ["television", "organism", "activity", "tool", "object", "device", "instrument", "abstraction", "instrumentality", "watch", "entity", "artifact", "person", "timepiece", "measure"]}, "book": {"RelatedTo": ["matter", "creation", "paper", "container", "group", "newspaper", "material", "whole", "object", "abstraction", "book", "entity", "product", "artifact"], "IsA": ["object", "device"], "AtLocation": ["bed"]}, "towel_holder": {"RelatedTo": ["towel_holder"]}, "furniture": {"RelatedTo": ["box", "lamp", "bed", "chair", "furniture", "cabinet"], "IsA": ["act", "artifact"], "AtLocation": ["room"]}, "cabinet": {"RelatedTo": ["room", "area", "food", "protective_covering", "structure", "whole", "physical_entity", "compartment", "instrumentality", "furniture", "cabinet", "artifact"]}, "tomato": {"RelatedTo": ["matter", "food", "organism", "living_thing", "whole", "herb", "vascular_plant", "physical_entity", "solanaceous_vegetable", "produce", "tomato", "apple"]}, "body_part": {"RelatedTo": ["mug", "body_part", "egg"]}, "tissue_box": {"RelatedTo": ["tissue_box"]}, "dirt": {"RelatedTo": ["matter", "substance", "material", "state", "physical_entity", "abstraction", "act", "condition", "dirt", "attribute"], "AtLocation": ["bed"]}, "knob": {"RelatedTo": ["structure", "shape", "whole", "convex_shape", "physical_entity", "solid", "knob", "artifact", "attribute"], "AtLocation": ["television"]}, "bathtub": {"RelatedTo": ["container", "whole", "physical_entity", "instrumentality", "bathtub", "artifact", "vessel"]}, "woody_plant": {"RelatedTo": ["box", "woody_plant", "apple"]}, "electronic_equipment": {"RelatedTo": ["television", "cellphone", "electronic_equipment"]}, "plunger": {"RelatedTo": ["organism", "implement", "living_thing", "whole", "tool", "device", "physical_entity", "instrumentality", "plunger", "artifact", "person"]}, "entity": {"RelatedTo": ["sandwich", "toaster", "omelette", "fried_egg", "watch", "book", "entity", "apple", "blinds", "egg"]}, "product": {"RelatedTo": ["newspaper", "material", "object", "book", "product"]}, "surface": {"RelatedTo": ["countertop", "bed", "surface"], "UsedFor": ["painting"], "AtLocation": ["mirror", "object", "solid"]}, "pen": {"RelatedTo": ["area", "paper", "structure", "writing_implement", "implement", "whole", "tool", "device", "physical_entity", "instrument", "instrumentality", "pen", "pencil", "artifact"], "IsA": ["object"], "AtLocation": ["box", "pen"]}, "cloth": {"RelatedTo": ["fabric", "material", "whole", "physical_entity", "cloth", "artifact"]}, "pencil": {"RelatedTo": ["matter", "substance", "writing_implement", "shape", "implement", "whole", "instrument", "abstraction", "instrumentality", "pen", "pencil", "artifact", "figure"], "AtLocation": ["cup"]}, "artifact": {"RelatedTo": ["laptop", "television", "countertop", "fridge", "box", "container", "food", "scrub_brush", "pot", "garbage_can", "sports_equipment", "group", "substance", "vacuum_cleaner", "mirror", "bowl", "microwave", "lamp", "newspaper", "cellphone", "mug", "alarm_clock", "fork", "cup", "pan", "plate", "watering_can", "towel", "toilet", "process", "painting", "bed", "chair", "statue", "remote_control", "watch", "book", "cabinet", "knob", "bathtub", "plunger", "pen", "cloth", "pencil", "artifact", "spoon", "sink", "person", "blinds", "pillow", "candle", "knife", "attribute", "hanger"], "IsA": ["food", "group", "substance", "shape", "object", "act", "artifact", "vessel", "art", "attribute"], "PartOf": ["group", "act", "artifact"], "UsedFor": ["creation", "food", "group", "substance", "state", "object", "process", "act", "artifact", "person", "attribute"], "AtLocation": ["object", "artifact"]}, "spoon": {"RelatedTo": ["tableware", "container", "whole", "fork", "physical_entity", "containerful", "cutlery", "abstraction", "instrumentality", "artifact", "spoon", "measure"], "AtLocation": ["bowl", "bed"]}, "cooking_utensil": {"RelatedTo": ["pot", "pan", "cooking_utensil"]}, "plumbing_fixture": {"RelatedTo": ["pot", "toilet", "plumbing_fixture", "sink"]}, "butterknife": {"RelatedTo": ["butterknife"]}, "apple": {"RelatedTo": ["matter", "food", "shape", "whole", "object", "physical_entity", "produce", "plant_organ", "tomato", "woody_plant", "entity", "apple"], "AtLocation": ["fridge"]}, "sink": {"RelatedTo": ["whole", "physical_entity", "process", "artifact", "plumbing_fixture", "sink", "vessel", "natural_depression"]}, "person": {"RelatedTo": ["toaster", "mug", "object", "chair", "watch", "plunger", "person", "hanger", "figure"], "IsA": ["group", "state", "person"], "PartOf": ["group", "object", "person"]}, "timepiece": {"RelatedTo": ["alarm_clock", "watch", "timepiece"]}, "table_top": {"RelatedTo": ["table_top"]}, "toilet_paper": {"RelatedTo": ["matter", "paper", "substance", "material", "physical_entity", "abstraction", "toilet_paper"], "AtLocation": ["toilet", "cabinet"]}, "measure": {"RelatedTo": ["box", "pot", "lettuce", "bowl", "mug", "cup", "plate", "bread", "watch", "spoon", "measure", "candle"]}, "vessel": {"RelatedTo": ["container", "pot", "bowl", "mug", "device", "bathtub", "sink", "vessel"], "IsA": ["artifact"]}, "art": {"RelatedTo": ["creation", "medium", "activity", "abstraction", "painting", "statue", "art"], "IsA": ["group", "state", "artifact", "person"]}, "blinds": {"RelatedTo": ["protective_covering", "group", "whole", "abstraction", "entity", "artifact", "blinds"]}, "bottlesoap": {"RelatedTo": ["bottlesoap"]}, "egg": {"RelatedTo": ["matter", "fridge", "container", "food", "foodstuff", "substance", "sandwich", "shape", "living_thing", "omelette", "whole", "object", "physical_entity", "produce", "body_part", "entity", "product", "person", "egg"], "AtLocation": ["fridge", "plate"]}, "pillow": {"RelatedTo": ["whole", "physical_entity", "bed", "artifact", "pillow"], "AtLocation": ["room"]}, "candle": {"RelatedTo": ["lamp", "whole", "device", "physical_entity", "instrumentality", "artifact", "measure", "candle"], "AtLocation": ["room", "cabinet"]}, "stove_burner": {"RelatedTo": ["stove_burner"]}, "home_appliance": {"RelatedTo": ["fridge", "vacuum_cleaner", "microwave", "toaster", "home_appliance"]}, "knife": {"RelatedTo": ["shape", "implement", "whole", "convex_shape", "tool", "device", "physical_entity", "instrument", "abstraction", "solid", "instrumentality", "artifact", "knife", "attribute"], "AtLocation": ["plate"]}, "attribute": {"RelatedTo": ["fork", "cup", "statue", "dirt", "knob", "knife", "attribute"], "IsA": ["shape", "state", "object", "act", "artifact", "attribute"], "PartOf": ["attribute"]}, "hanger": {"RelatedTo": ["organism", "living_thing", "whole", "device", "support", "physical_entity", "instrumentality", "artifact", "person", "hanger"]}, "natural_depression": {"RelatedTo": ["bed", "sink", "natural_depression"]}, "figure": {"RelatedTo": ["shape", "fork", "statue", "pencil", "person", "figure"]}};

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
	var nodes = toVisualize[0];
	var links = toVisualize[1];
	var width = 2000 //window.innerWidth
	var height = 2000//window.innerHeight
	// var width = window.innerWidth
	// var height = window.innerHeight

	var svg = d3.select('svg')
	svg.attr('width', width).attr('height', height)

	// simulation setup with all forces
	var linkForce = d3
	  .forceLink()
	  .id(function (link) { return link.id })
	  .strength(function (link) { return link.strength })

	var simulation = d3
	  .forceSimulation()
	  .force('link', linkForce)
	  .force('charge', d3.forceManyBody().strength(-10000))
	  .force('center', d3.forceCenter(width / 2, height / 2))

	var linkElements = svg.append("g")
	  .attr("class", "links")
	  .selectAll("line")
	  .data(links)
	  .enter().append("line")
	    .attr("stroke-width", 1)
		  .attr('stroke', function (link) { return getLinkColor(link) })

	var nodeElements = svg.append("g")
	  .attr("class", "nodes")
	  .selectAll("circle")
	  .data(nodes)
	  .enter().append("circle")
	    .attr("r", 10)
	    .attr("fill", 'grey')

	var textElements = svg.append("g")
	  .attr("class", "texts")
	  .selectAll("text")
	  .data(nodes)
	  .enter().append("text")
	    .text(function (node) { return  node.label })
		  .attr("font-size", 15)
		  .attr("dx", 15)
	    .attr("dy", 4)

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

	simulation.force("link").links(links)

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