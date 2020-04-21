# Use the following command to run clustering
# Rscript --vanilla clustering_BIC.R path/to/tagger/output path/to/index/output
# example Rscript --vanilla clustering_BIC.R result_epoch_4.json biased_token_idx.json

require(mclust)
require(rjson)
# choose optimal number of clusters

clustering = function(x) {
  # input: x, a vector of probability
  # return: idx, indices of clusters with highest probabilities
  clust = densityMclust(x)
  ?densityMclust
  # plot(clust)
  ## obtain cluster idx of max probability
  idx_max = which.max(x)
  cluster_id = clust$classification[idx_max]
  ## obtain idx of probability from above cluster
  indices = seq_along(x)[clust$classification==cluster_id]
  ## choose consequtive idx left and right of max probability
  idx_result = c()
  # left pass
  for(i in idx_max:1) {
    if(i %in% indices)
      idx_result = c(i,idx_result)
    else
      break
  }
  # right pass
  for(i in idx_max:length(x)) {
    if(i %in% indices)
      idx_result = c(idx_result,i)
    else
      break
  }
  unique(idx_result)
}


# x = c(0.14291854202747345, 0.17621269822120667, 0.31127846240997314, 0.4213598668575287, 0.2846294939517975, 0.2442476898431778, 0.13336563110351562, 0.15951526165008545, 0.13239271938800812, 0.12831713259220123, 0.12174572050571442, 0.126440167427063, 0.08297944813966751, 0.10982384532690048, 0.14110948145389557, 0.10982384532690048, 0.13069333136081696, 0.08226603269577026, 0.08424215763807297, 0.11979220807552338, 0.06421235203742981, 0.10544000566005707, 0.14580751955509186, 0.22787918150424957, 0.23591408133506775, 0.17235904932022095, 0.17916880548000336, 0.08226603269577026, 0.1141601949930191, 0.20096933841705322, 0.20405332744121552, 0.3101278841495514, 0.17621268332004547, 0.214020773768425, 0.1170048862695694, 0.1264401227235794, 0.12024607509374619, 0.06906124949455261, 0.10183805227279663, 0.1025921180844307, 0.20446951687335968, 0.1227477490901947, 0.11443036049604416, 0.15967343747615814, 0.21044333279132843, 0.11449217796325684, 0.12802408635616302, 0.13908427953720093, 0.1376064568758011, 0.16198179125785828, 0.19585932791233063, 0.4059872627258301, 0.20629672706127167, 0.40319058299064636, 0.12080153822898865, 0.29184490442276, 0.10183805227279663, 0.13703303039073944, 0.162053182721138, 0.162053182721138, 0.15109975636005402, 0.18323980271816254, 0.25257018208503723, 0.204423189163208, 0.14826631546020508, 0.1979716420173645, 0.18501046299934387, 0.11669707298278809)
# # 
# clust_idx = clustering(x)
# clust_idx
# 
# x[clust_idx]
# plot(clust)
# clust$BIC

# read json file
print("Reading json file...")
args = commandArgs(trailingOnly=TRUE)
# args = '/Users/roib/Downloads/neutralizing-bias/src/result_epoch_4.json'
tagger_output <- fromJSON(file=args)
# calculate clustering index for each probability vector
print('Calculate idx of cluster with highest probability...')
idx_list = lapply(tagger_output$tok_probs, function(x) clustering(x))
# Write output to JSON
print("Write result to json...")
jsonData = toJSON(idx_list)
# write(jsonData, "/Users/roib/Downloads/output.json") 
write(jsonData, "biased_token_idx.json")
