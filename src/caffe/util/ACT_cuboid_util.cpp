#include <algorithm>
#include <csignal>
#include <ctime>
#include <functional>
#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "boost/iterator/counting_iterator.hpp"

// ACT-detector: use functions from bbox_util of SSD 
#include "caffe/util/bbox_util.hpp"
#include "caffe/util/ACT_cuboid_util.hpp"

namespace caffe {

template <typename Dtype>
void ACTGetGroundTruth(const Dtype* gt_data, const int num_gt,
      const int background_label_id, const bool use_difficult_gt,
      map<int, vector<ACTNormalizedTube> >* all_gt_tubes, const int sequence_length) {

  all_gt_tubes->clear();
  for (int i = 0; i < num_gt; ++i) {
    int start_idx = i * sequence_length * 8;

    int item_id = gt_data[start_idx];
    if (item_id == -1) {
      continue;
    }
    
    int label = gt_data[start_idx + 1];
    CHECK_NE(background_label_id, label)
        << "Found background label in the dataset.";
    bool difficult = static_cast<bool>(gt_data[start_idx + 7]);
    if (!use_difficult_gt && difficult) {
      // Skip reading difficult ground truth.
      continue;
    }

    ACTNormalizedTube tube;
    // ACT-detector: for all boxes in the sequence
    for( int j = 0 ; j < sequence_length ; ++j  ){
      NormalizedBBox bbox;
      bbox.set_label(label);
      // ACT-detector: the gt tube should have one constant label.. 
      CHECK_EQ( label, gt_data[start_idx +1]) << "ACT-detector: the label of the ground truth tube changes across time";
      bbox.set_xmin(gt_data[start_idx + 3]);
      bbox.set_ymin(gt_data[start_idx + 4]);
      bbox.set_xmax(gt_data[start_idx + 5]);
      bbox.set_ymax(gt_data[start_idx + 6]);
      bbox.set_difficult(difficult);
      float bbox_size = BBoxSize(bbox);
      bbox.set_size(bbox_size);
      tube.push_back(bbox);
      // ACT-detector: increase the counter for every tube 
      start_idx+=8; 
    }

    (*all_gt_tubes)[item_id].push_back(tube);

  }
}

// explicit initialization
template void ACTGetGroundTruth(const float* gt_data, const int num_gt,
      const int background_label_id, const bool use_difficult_gt,
      map<int, vector<ACTNormalizedTube> >* all_gt_tubes, const int sequence_length);
template void ACTGetGroundTruth(const double* gt_data, const int num_gt,
      const int background_label_id, const bool use_difficult_gt,
      map<int, vector<ACTNormalizedTube> >* all_gt_tubes, const int sequence_length);





template <typename Dtype>
void ACTGetPriorTubes(const Dtype* prior_data, const int num_priors,
      vector<ACTNormalizedTube>* prior_tubes,
      vector<vector<float> >* prior_variances,
      const int sequence_length) {
        
  prior_tubes->clear();
  prior_variances->clear();
  
  for (int i = 0; i < num_priors; ++i) {
    int start_idx = i * 4;
    NormalizedBBox bbox;
    bbox.set_xmin(prior_data[start_idx]);
    bbox.set_ymin(prior_data[start_idx + 1]);
    bbox.set_xmax(prior_data[start_idx + 2]);
    bbox.set_ymax(prior_data[start_idx + 3]);
    float bbox_size = BBoxSize(bbox);
    bbox.set_size(bbox_size);

    ACTNormalizedTube tube;
    // ACT-detector: for all boxes in the sequence
    for (int j = 0 ; j < sequence_length ; ++j ){
      tube.push_back(bbox);
    }
    prior_tubes->push_back(tube);
    
  }

  for (int i = 0; i < num_priors; ++i) {
    int start_idx = (num_priors + i) * 4;
    vector<float> var;
    for (int j = 0; j < 4; ++j) {
      var.push_back(prior_data[start_idx + j]);
    }
    prior_variances->push_back(var);
  }

}

// Explicit initialization.
template void ACTGetPriorTubes(const float* prior_data, const int num_priors,
      vector<ACTNormalizedTube>* prior_tubes,
      vector<vector<float> >* prior_variances,
      const int sequence_length);
template void ACTGetPriorTubes(const double* prior_data, const int num_priors,
      vector<ACTNormalizedTube>* prior_tubes,
      vector<vector<float> >* prior_variances,
      const int sequence_length);





template <typename Dtype>
void ACTGetLocPredictions(const Dtype* loc_data, const int num,
      const int num_preds_per_class, const int num_loc_classes,
      const bool share_location, vector<ACTLabelTube>* loc_preds,
      const int sequence_length) {

  loc_preds->clear();
  if (share_location) {
    CHECK_EQ(num_loc_classes, 1);
  }

  loc_preds->resize(num);

  for (int i = 0; i < num; ++i) {
    
    ACTLabelTube& label_tube = (*loc_preds)[i];
    
    for (int p = 0; p < num_preds_per_class; ++p) {
      int start_idx = p * num_loc_classes * 4 * sequence_length;
      for (int c = 0; c < num_loc_classes; ++c) {
        int label = share_location ? -1 : c;
        if (label_tube.find(label) == label_tube.end()) {
          label_tube[label].resize(num_preds_per_class);
        }

        label_tube[label][p].resize(sequence_length);
        for(int j=0 ; j<sequence_length ; j++){
          label_tube[label][p][j].set_xmin( loc_data[start_idx + (c*sequence_length+j)*4 ]);
          label_tube[label][p][j].set_ymin( loc_data[start_idx + (c*sequence_length+j)*4 + 1]);
          label_tube[label][p][j].set_xmax( loc_data[start_idx + (c*sequence_length+j)*4 + 2]);
          label_tube[label][p][j].set_ymax( loc_data[start_idx + (c*sequence_length+j)*4 + 3]);
        }
      }
    }
    loc_data += num_preds_per_class * num_loc_classes * 4 * sequence_length;
  }
}

// Explicit initialization.
template void ACTGetLocPredictions(const float* loc_data, const int num,
      const int num_preds_per_class, const int num_loc_classes,
      const bool share_location, vector<ACTLabelTube>* loc_preds,
      const int sequence_length);
template void ACTGetLocPredictions(const double* loc_data, const int num,
      const int num_preds_per_class, const int num_loc_classes,
      const bool share_location, vector<ACTLabelTube>* loc_preds,
      const int sequence_length);





bool IsCrossBoundaryTube(const ACTNormalizedTube& tube) {
  bool ret = false;
  for ( int j = 0 ; j < tube.size() ; j++ ){
    if ( tube[j].xmin() < 0 || tube[j].xmin() > 1 ||
         tube[j].ymin() < 0 || tube[j].ymin() > 1 ||
         tube[j].xmax() < 0 || tube[j].xmax() > 1 ||
         tube[j].ymax() < 0 || tube[j].ymax() > 1 ) {
        ret = true;
      }
  }
  return ret;
}

// ACT-detector: jaccard overlap for tubes 
float JaccardOverlapTube(const ACTNormalizedTube& tube1, const ACTNormalizedTube& tube2, 
                         const bool normalized) {
  float jaccard_overlap = 0.0;
  CHECK_EQ( tube1.size(), tube2.size() ) << "ACT-detector: tubes must have the same size";
  for( int j = 0 ;  j < tube1.size() ; j++ ){
    jaccard_overlap += JaccardOverlap( tube1[j], tube2[j], normalized );
  }
  return jaccard_overlap / tube1.size();
}

// ACT-detector: used in the ACT_detection_output_layer: ACTApplyNMSTubeFast
template <typename Dtype> 
Dtype JaccardOverlapTube(const Dtype* tube1, const Dtype* tube2, 
                         const int sequence_length){
    float jaccard_overlap = 0.0; 
    for ( int j = 0;  j < sequence_length; j++ ){
        jaccard_overlap += JaccardOverlap( tube1 + 4*j, tube2 + 4*j );
    }
    return jaccard_overlap / sequence_length;
}
template float JaccardOverlapTube(const float* tube1, const float* tube2, 
                                  const int sequence_length);
template double JaccardOverlapTube(const double* tube1, const double* tube2, 
                                   const int sequence_length);


void ACTMatchTube(const vector<ACTNormalizedTube>& gt_tubes,
    const vector<ACTNormalizedTube>& pred_tubes, const int label,
    const MatchType match_type, const float overlap_threshold,
    const bool ignore_cross_boundary_bbox,
    vector<int>* match_indices, vector<float>* match_overlaps) {
  int num_pred = pred_tubes.size();
  match_indices->clear();
  match_indices->resize(num_pred, -1);
  match_overlaps->clear();
  match_overlaps->resize(num_pred, 0.);

  int num_gt = 0;
  vector<int> gt_indices;
  if (label == -1) {
    // label -1 means comparing against all ground truth.
    num_gt = gt_tubes.size();
    for (int i = 0; i < num_gt; ++i) {
      gt_indices.push_back(i);
    }
  } else {
    // ACT-detector: count number of ground truth boxes which have the desired label.
    for (int i = 0; i < gt_tubes.size(); ++i) {
      // ACT-detector: the label is constant over the whole sequence -> use only the one of the 1st frame
      if (gt_tubes[i][0].label() == label) {  
        num_gt++;
        gt_indices.push_back(i);
      }
    }
  }
  if (num_gt == 0) {
    return;
  }

  // Store the positive overlap between predictions and ground truth.
  map<int, map<int, float> > overlaps;
  for (int i = 0; i < num_pred; ++i) {
    if (ignore_cross_boundary_bbox && IsCrossBoundaryTube(pred_tubes[i])) {
      (*match_indices)[i] = -2;
      continue;
    }
    for (int j = 0; j < num_gt; ++j) {
      float overlap = JaccardOverlapTube(pred_tubes[i], gt_tubes[gt_indices[j]]);
      if (overlap > 1e-6) {
        (*match_overlaps)[i] = std::max((*match_overlaps)[i], overlap);
        overlaps[i][j] = overlap;
      }
    }
  }

  // Bipartite matching.
  vector<int> gt_pool;
  for (int i = 0; i < num_gt; ++i) {
    gt_pool.push_back(i);
  }
  while (gt_pool.size() > 0) {
    // Find the most overlapped gt and cooresponding predictions.
    int max_idx = -1;
    int max_gt_idx = -1;
    float max_overlap = -1;
    for (map<int, map<int, float> >::iterator it = overlaps.begin();
         it != overlaps.end(); ++it) {
      int i = it->first;
      if ((*match_indices)[i] != -1) {
        // The prediction already has matched ground truth or is ignored.
        continue;
      }
      for (int p = 0; p < gt_pool.size(); ++p) {
        int j = gt_pool[p];
        if (it->second.find(j) == it->second.end()) {
          // No overlap between the i-th prediction and j-th ground truth.
          continue;
        }
        // Find the maximum overlapped pair.
        if (it->second[j] > max_overlap) {
          // If the prediction has not been matched to any ground truth,
          // and the overlap is larger than maximum overlap, update.
          max_idx = i;
          max_gt_idx = j;
          max_overlap = it->second[j];
        }
      }
    }
    if (max_idx == -1) {
      // Cannot find good match.
      break;
    } else {
      CHECK_EQ((*match_indices)[max_idx], -1);
      (*match_indices)[max_idx] = gt_indices[max_gt_idx];
      (*match_overlaps)[max_idx] = max_overlap;
      // Erase the ground truth.
      gt_pool.erase(std::find(gt_pool.begin(), gt_pool.end(), max_gt_idx));
    }
  }

  switch (match_type) {
    case MultiBoxLossParameter_MatchType_BIPARTITE:
      // Already done.
      break;
    case MultiBoxLossParameter_MatchType_PER_PREDICTION:
      // Get most overlaped for the rest prediction tubes.
      for (map<int, map<int, float> >::iterator it = overlaps.begin();
           it != overlaps.end(); ++it) {
        int i = it->first;
        if ((*match_indices)[i] != -1) {
          // The prediction already has matched ground truth or is ignored.
          continue;
        }
        int max_gt_idx = -1;
        float max_overlap = -1;
        for (int j = 0; j < num_gt; ++j) {
          if (it->second.find(j) == it->second.end()) {
            // No overlap between the i-th prediction and j-th ground truth.
            continue;
          }
          // Find the maximum overlapped pair.
          float overlap = it->second[j];
          if (overlap >= overlap_threshold && overlap > max_overlap) {
            // If the prediction has not been matched to any ground truth,
            // and the overlap is larger than maximum overlap, update.
            max_gt_idx = j;
            max_overlap = overlap;
          }
        }
        if (max_gt_idx != -1) {
          // Found a matched ground truth.
          CHECK_EQ((*match_indices)[i], -1);
          (*match_indices)[i] = gt_indices[max_gt_idx];
          (*match_overlaps)[i] = max_overlap;
        }
      }
      break;
    default:
      LOG(FATAL) << "Unknown matching type.";
      break;
  }

  return;
}


void ACTFindMatches(const vector<ACTLabelTube>& all_loc_preds,
      const map<int, vector<ACTNormalizedTube> >& all_gt_tubes,
      const vector<ACTNormalizedTube>& prior_tubes,
      const vector<vector<float> >& prior_variances,
      const MultiBoxLossParameter& multibox_loss_param,
      vector<map<int, vector<float> > >* all_match_overlaps,
      vector<map<int, vector<int> > >* all_match_indices) {
  // all_match_overlaps->clear();
  // all_match_indices->clear();
  // Get parameters.
  CHECK(multibox_loss_param.has_num_classes()) << "Must provide num_classes.";
  const int num_classes = multibox_loss_param.num_classes();
  CHECK_GE(num_classes, 1) << "num_classes should not be less than 1.";
  const bool share_location = multibox_loss_param.share_location();
  const int loc_classes = share_location ? 1 : num_classes;
  const MatchType match_type = multibox_loss_param.match_type();
  const float overlap_threshold = multibox_loss_param.overlap_threshold();
  const bool use_prior_for_matching =
      multibox_loss_param.use_prior_for_matching();
      
  CHECK( use_prior_for_matching ) << "ACT-detector: Not implemented";
  CHECK_EQ( match_type, MultiBoxLossParameter_MatchType_PER_PREDICTION ) << "ACT-detector: Not implemented";
  
  const int background_label_id = multibox_loss_param.background_label_id();  
  const bool ignore_cross_boundary_bbox =
      multibox_loss_param.ignore_cross_boundary_bbox();
  // Find the matches.
  int num = all_loc_preds.size();
  for (int i = 0; i < num; ++i) {
    map<int, vector<int> > match_indices;
    map<int, vector<float> > match_overlaps;
    // Check if there is ground truth for current image.
    if (all_gt_tubes.find(i) == all_gt_tubes.end()) {
      // There is no gt for current image. All predictions are negative.
      all_match_indices->push_back(match_indices);
      all_match_overlaps->push_back(match_overlaps);
      continue;
    }
    // Find match between predictions and ground truth.
    const vector<ACTNormalizedTube>& gt_tubes = all_gt_tubes.find(i)->second;
    if (!use_prior_for_matching) {
      // ACT-detector: not implemented      
    } else {
      // ACT-detector: use prior tubes to match against all ground truth tubes.
      vector<int> temp_match_indices;
      vector<float> temp_match_overlaps;
      const int label = -1;
      ACTMatchTube(gt_tubes, prior_tubes, label, match_type, overlap_threshold,
                ignore_cross_boundary_bbox, &temp_match_indices,
                &temp_match_overlaps);

      if (share_location) {
        match_indices[label] = temp_match_indices;
        match_overlaps[label] = temp_match_overlaps;
      } else {
        // ACT-detector: get ground truth label for each ground truth tube.
        vector<int> gt_labels;
        for (int g = 0; g < gt_tubes.size(); ++g) {
          // ACT-detector: the label is constant across the tube --> get the label of the first frame only
          gt_labels.push_back(gt_tubes[g][0].label()); 
        }
        // Distribute the matching results to different loc_class.
        for (int c = 0; c < loc_classes; ++c) {
          if (c == background_label_id) {
            // Ignore background loc predictions.
            continue;
          }
          match_indices[c].resize(temp_match_indices.size(), -1);
          match_overlaps[c] = temp_match_overlaps;
          for (int m = 0; m < temp_match_indices.size(); ++m) {
            if (temp_match_indices[m] > -1) {
              const int gt_idx = temp_match_indices[m];
              CHECK_LT(gt_idx, gt_labels.size());
              if (c == gt_labels[gt_idx]) {
                match_indices[c][m] = gt_idx;
              }
            }
          }
        }
      }
    }
    all_match_indices->push_back(match_indices);
    all_match_overlaps->push_back(match_overlaps);
  }
}


template <typename Dtype>
void ACTComputeConfLoss(const Dtype* conf_data, const int num,
      const int num_preds_per_class, const int num_classes,
      const int background_label_id, const ConfLossType loss_type,
      const vector<map<int, vector<int> > >& all_match_indices,
      const map<int, vector<ACTNormalizedTube> >& all_gt_tubes,
      vector<vector<float> >* all_conf_loss) {
  CHECK_LT(background_label_id, num_classes);
  all_conf_loss->clear();
  for (int i = 0; i < num; ++i) {
    vector<float> conf_loss;
    const map<int, vector<int> >& match_indices = all_match_indices[i];
    for (int p = 0; p < num_preds_per_class; ++p) {
      int start_idx = p * num_classes;
      // Get the label index.
      int label = background_label_id;
      for (map<int, vector<int> >::const_iterator it =
           match_indices.begin(); it != match_indices.end(); ++it) {
        const vector<int>& match_index = it->second;
        CHECK_EQ(match_index.size(), num_preds_per_class);
        if (match_index[p] > -1) {
          CHECK(all_gt_tubes.find(i) != all_gt_tubes.end());
          const vector<ACTNormalizedTube>& gt_tubes =
              all_gt_tubes.find(i)->second;
          CHECK_LT(match_index[p], gt_tubes.size());
          // ACT-detector: parse the label of the tube by looking at the label of the first frame, 
          // as the label of the tube is constant across its frames
          label = gt_tubes[match_index[p]][0].label(); 
          CHECK_GE(label, 0);
          CHECK_NE(label, background_label_id);
          CHECK_LT(label, num_classes);
          // A prior can only be matched to one gt bbox.
          break;
        }
      }
      Dtype loss = 0;
      if (loss_type == MultiBoxLossParameter_ConfLossType_SOFTMAX) {
        CHECK_GE(label, 0);
        CHECK_LT(label, num_classes);
        // Compute softmax probability.
        // We need to subtract the max to avoid numerical issues.
        Dtype maxval = conf_data[start_idx];
        for (int c = 1; c < num_classes; ++c) {
          maxval = std::max<Dtype>(conf_data[start_idx + c], maxval);
        }
        Dtype sum = 0.;
        for (int c = 0; c < num_classes; ++c) {
          sum += std::exp(conf_data[start_idx + c] - maxval);
        }
        Dtype prob = std::exp(conf_data[start_idx + label] - maxval) / sum;
        loss = -log(std::max(prob, Dtype(FLT_MIN)));
      } else if (loss_type == MultiBoxLossParameter_ConfLossType_LOGISTIC) {
        int target = 0;
        for (int c = 0; c < num_classes; ++c) {
          if (c == label) {
            target = 1;
          } else {
            target = 0;
          }
          Dtype input = conf_data[start_idx + c];
          loss -= input * (target - (input >= 0)) -
              log(1 + exp(input - 2 * input * (input >= 0)));
        }
      } else {
        LOG(FATAL) << "Unknown conf loss type.";
      }
      conf_loss.push_back(loss);
    }
    conf_data += num_preds_per_class * num_classes;
    all_conf_loss->push_back(conf_loss);
  }
}

// Explicit initialization.
template void ACTComputeConfLoss(const float* conf_data, const int num,
      const int num_preds_per_class, const int num_classes,
      const int background_label_id, const ConfLossType loss_type,
      const vector<map<int, vector<int> > >& all_match_indices,
      const map<int, vector<ACTNormalizedTube> >& all_gt_tubes,
      vector<vector<float> >* all_conf_loss);
template void ACTComputeConfLoss(const double* conf_data, const int num,
      const int num_preds_per_class, const int num_classes,
      const int background_label_id, const ConfLossType loss_type,
      const vector<map<int, vector<int> > >& all_match_indices,
      const map<int, vector<ACTNormalizedTube> >& all_gt_tubes,
      vector<vector<float> >* all_conf_loss);

inline bool IsEligibleMining(const MiningType mining_type, const int match_idx,
    const float match_overlap, const float neg_overlap) {
  if (mining_type == MultiBoxLossParameter_MiningType_MAX_NEGATIVE) {
    return match_idx == -1 && match_overlap < neg_overlap;
  } else if (mining_type == MultiBoxLossParameter_MiningType_HARD_EXAMPLE) {
    return true;
  } else {
    return false;
  }
}

template <typename Dtype>
void ACTMineHardExamples(const Blob<Dtype>& conf_blob,
    const vector<ACTLabelTube>& all_loc_preds,
    const map<int, vector<ACTNormalizedTube> >& all_gt_tubes,
    const vector<ACTNormalizedTube>& prior_tubes,
    const vector<vector<float> >& prior_variances,
    const vector<map<int, vector<float> > >& all_match_overlaps,
    const MultiBoxLossParameter& multibox_loss_param,
    int* num_matches, int* num_negs,
    vector<map<int, vector<int> > >* all_match_indices,
    vector<vector<int> >* all_neg_indices) {
  int num = all_loc_preds.size();
  // CHECK_EQ(num, all_match_overlaps.size());
  // CHECK_EQ(num, all_match_indices->size());
  // all_neg_indices->clear();
  *num_matches = CountNumMatches(*all_match_indices, num);
  *num_negs = 0;
  int num_priors = prior_tubes.size();
  CHECK_EQ(num_priors, prior_variances.size());
  // Get parameters.
  CHECK(multibox_loss_param.has_num_classes()) << "Must provide num_classes.";
  const int num_classes = multibox_loss_param.num_classes();
  CHECK_GE(num_classes, 1) << "num_classes should not be less than 1.";
  const int background_label_id = multibox_loss_param.background_label_id();
  // const bool use_prior_for_nms = multibox_loss_param.use_prior_for_nms();
  const ConfLossType conf_loss_type = multibox_loss_param.conf_loss_type();
  const MiningType mining_type = multibox_loss_param.mining_type();
  if (mining_type == MultiBoxLossParameter_MiningType_NONE) {
    return;
  }
  CHECK( mining_type!=MultiBoxLossParameter_MiningType_HARD_EXAMPLE ) << "ACT-detector: Not implemented";  
  const float neg_pos_ratio = multibox_loss_param.neg_pos_ratio();
  const float neg_overlap = multibox_loss_param.neg_overlap();  
  const bool has_nms_param = multibox_loss_param.has_nms_param();
  float nms_threshold = 0;
  if (has_nms_param) {
    nms_threshold = multibox_loss_param.nms_param().nms_threshold();
  }
  const int sample_size = multibox_loss_param.sample_size();
  // Compute confidence losses based on matching results.
  vector<vector<float> > all_conf_loss;
#ifdef CPU_ONLY
  ACTComputeConfLoss(conf_blob.cpu_data(), num, num_priors, num_classes,
      background_label_id, conf_loss_type, *all_match_indices, all_gt_tubes,
      &all_conf_loss);
#else
  ACTComputeConfLossGPU(conf_blob, num, num_priors, num_classes,
      background_label_id, conf_loss_type, *all_match_indices, all_gt_tubes,
      &all_conf_loss);
#endif

  vector<vector<float> > all_loc_loss;
  if (mining_type == MultiBoxLossParameter_MiningType_HARD_EXAMPLE) {
    CHECK(false) << "ACT-detector: Not implemented";
  } else {
    // No localization loss.
    for (int i = 0; i < num; ++i) {
      vector<float> loc_loss(num_priors, 0.f);
      all_loc_loss.push_back(loc_loss);
    }
  }
  for (int i = 0; i < num; ++i) {
    map<int, vector<int> >& match_indices = (*all_match_indices)[i];
    const map<int, vector<float> >& match_overlaps = all_match_overlaps[i];
    // loc + conf loss.
    const vector<float>& conf_loss = all_conf_loss[i];
    const vector<float>& loc_loss = all_loc_loss[i];
    vector<float> loss;
    std::transform(conf_loss.begin(), conf_loss.end(), loc_loss.begin(),
                   std::back_inserter(loss), std::plus<float>());
    // Pick negatives or hard examples based on loss.
    set<int> sel_indices;
    vector<int> neg_indices;
    for (map<int, vector<int> >::iterator it = match_indices.begin();
         it != match_indices.end(); ++it) {
      const int label = it->first;
      int num_sel = 0;
      // Get potential indices and loss pairs.
      vector<pair<float, int> > loss_indices;
      for (int m = 0; m < match_indices[label].size(); ++m) {
        if (IsEligibleMining(mining_type, match_indices[label][m],
            match_overlaps.find(label)->second[m], neg_overlap)) {
          loss_indices.push_back(std::make_pair(loss[m], m));
          ++num_sel;
        }
      }
      if (mining_type == MultiBoxLossParameter_MiningType_MAX_NEGATIVE) {
        int num_pos = 0;
        for (int m = 0; m < match_indices[label].size(); ++m) {
          if (match_indices[label][m] > -1) {
            ++num_pos;
          }
        }
        num_sel = std::min(static_cast<int>(num_pos * neg_pos_ratio), num_sel);
      } else if (mining_type == MultiBoxLossParameter_MiningType_HARD_EXAMPLE) {
        CHECK_GT(sample_size, 0);
        num_sel = std::min(sample_size, num_sel);
      }
      // Select samples.
      if (has_nms_param && nms_threshold > 0) {
        CHECK(false) << "ACT-detector: Not implemented";        
      } else {
        // Pick top example indices based on loss.
        std::sort(loss_indices.begin(), loss_indices.end(),
                  SortScorePairDescend<int>);
        for (int n = 0; n < num_sel; ++n) {
          sel_indices.insert(loss_indices[n].second);
        }
      }
      // Update the match_indices and select neg_indices.
      for (int m = 0; m < match_indices[label].size(); ++m) {
        if (match_indices[label][m] > -1) {
          if (mining_type == MultiBoxLossParameter_MiningType_HARD_EXAMPLE &&
              sel_indices.find(m) == sel_indices.end()) {
            match_indices[label][m] = -1;
            *num_matches -= 1;
          }
        } else if (match_indices[label][m] == -1) {
          if (sel_indices.find(m) != sel_indices.end()) {
            neg_indices.push_back(m);
            *num_negs += 1;
          }
        }
      }
    }
    all_neg_indices->push_back(neg_indices);
  }
}

// Explicite initialization.
template void ACTMineHardExamples(const Blob<float>& conf_blob,
    const vector<ACTLabelTube>& all_loc_preds,
    const map<int, vector<ACTNormalizedTube> >& all_gt_tubes,
    const vector<ACTNormalizedTube>& prior_tubes,
    const vector<vector<float> >& prior_variances,
    const vector<map<int, vector<float> > >& all_match_overlaps,
    const MultiBoxLossParameter& multibox_loss_param,
    int* num_matches, int* num_negs,
    vector<map<int, vector<int> > >* all_match_indices,
    vector<vector<int> >* all_neg_indices);

template void ACTMineHardExamples(const Blob<double>& conf_blob,
    const vector<ACTLabelTube>& all_loc_preds,
    const map<int, vector<ACTNormalizedTube> >& all_gt_tubes,
    const vector<ACTNormalizedTube>& prior_tubes,
    const vector<vector<float> >& prior_variances,
    const vector<map<int, vector<float> > >& all_match_overlaps,
    const MultiBoxLossParameter& multibox_loss_param,
    int* num_matches, int* num_negs,
    vector<map<int, vector<int> > >* all_match_indices,
    vector<vector<int> >* all_neg_indices);

template <typename Dtype>
void ACTEncodeConfPrediction(const Dtype* conf_data, const int num,
      const int num_priors, const MultiBoxLossParameter& multibox_loss_param,
      const vector<map<int, vector<int> > >& all_match_indices,
      const vector<vector<int> >& all_neg_indices,
      const map<int, vector<ACTNormalizedTube> >& all_gt_tubes,
      Dtype* conf_pred_data, Dtype* conf_gt_data) {
  // CHECK_EQ(num, all_match_indices.size());
  // CHECK_EQ(num, all_neg_indices.size());
  // Retrieve parameters.
  CHECK(multibox_loss_param.has_num_classes()) << "Must provide num_classes.";
  const int num_classes = multibox_loss_param.num_classes();
  CHECK_GE(num_classes, 1) << "num_classes should not be less than 1.";
  const int background_label_id = multibox_loss_param.background_label_id();
  const bool map_object_to_agnostic =
      multibox_loss_param.map_object_to_agnostic();
  if (map_object_to_agnostic) {
    if (background_label_id >= 0) {
      CHECK_EQ(num_classes, 2);
    } else {
      CHECK_EQ(num_classes, 1);
    }
  }
  const MiningType mining_type = multibox_loss_param.mining_type();
  bool do_neg_mining;
  if (multibox_loss_param.has_do_neg_mining()) {
    LOG(WARNING) << "do_neg_mining is deprecated, use mining_type instead.";
    do_neg_mining = multibox_loss_param.do_neg_mining();
    CHECK_EQ(do_neg_mining,
             mining_type != MultiBoxLossParameter_MiningType_NONE);
  }
  do_neg_mining = mining_type != MultiBoxLossParameter_MiningType_NONE;
  const ConfLossType conf_loss_type = multibox_loss_param.conf_loss_type();
  int count = 0;
  for (int i = 0; i < num; ++i) {
    if (all_gt_tubes.find(i) != all_gt_tubes.end()) {
      // Save matched (positive) bboxes scores and labels.
      const map<int, vector<int> >& match_indices = all_match_indices[i];
      for (map<int, vector<int> >::const_iterator it =
          match_indices.begin(); it != match_indices.end(); ++it) {
        const vector<int>& match_index = it->second;
        CHECK_EQ(match_index.size(), num_priors);
        for (int j = 0; j < num_priors; ++j) {
          if (match_index[j] <= -1) {
            continue;
          }
          const int gt_label = map_object_to_agnostic ?
            background_label_id + 1 :
            // ACT-detector: access label of a tube by accessing the label of the first frame of the sequence of frames
            all_gt_tubes.find(i)->second[match_index[j]][0].label(); 
          int idx = do_neg_mining ? count : j;
          switch (conf_loss_type) {
            case MultiBoxLossParameter_ConfLossType_SOFTMAX:
              conf_gt_data[idx] = gt_label;
              break;
            case MultiBoxLossParameter_ConfLossType_LOGISTIC:
              conf_gt_data[idx * num_classes + gt_label] = 1;
              break;
            default:
              LOG(FATAL) << "Unknown conf loss type.";
          }
          if (do_neg_mining) {
            // Copy scores for matched tubes.
            caffe_copy<Dtype>(num_classes, conf_data + j * num_classes,
                conf_pred_data + count * num_classes);
            ++count;
          }
        }
      }
      // Go to next image.
      if (do_neg_mining) {
        // Save negative tubes scores and labels.
        for (int n = 0; n < all_neg_indices[i].size(); ++n) {
          int j = all_neg_indices[i][n];
          CHECK_LT(j, num_priors);
          caffe_copy<Dtype>(num_classes, conf_data + j * num_classes,
              conf_pred_data + count * num_classes);
          switch (conf_loss_type) {
            case MultiBoxLossParameter_ConfLossType_SOFTMAX:
              conf_gt_data[count] = background_label_id;
              break;
            case MultiBoxLossParameter_ConfLossType_LOGISTIC:
              if (background_label_id >= 0 &&
                  background_label_id < num_classes) {
                conf_gt_data[count * num_classes + background_label_id] = 1;
              }
              break;
            default:
              LOG(FATAL) << "Unknown conf loss type.";
          }
          ++count;
        }
      }
    }
    if (do_neg_mining) {
      conf_data += num_priors * num_classes;
    } else {
      conf_gt_data += num_priors;
    }
  }
}

// Explicite initialization.
template void ACTEncodeConfPrediction(const float* conf_data, const int num,
      const int num_priors, const MultiBoxLossParameter& multibox_loss_param,
      const vector<map<int, vector<int> > >& all_match_indices,
      const vector<vector<int> >& all_neg_indices,
      const map<int, vector<ACTNormalizedTube> >& all_gt_tubes,
      float* conf_pred_data, float* conf_gt_data);
template void ACTEncodeConfPrediction(const double* conf_data, const int num,
      const int num_priors, const MultiBoxLossParameter& multibox_loss_param,
      const vector<map<int, vector<int> > >& all_match_indices,
      const vector<vector<int> >& all_neg_indices,
      const map<int, vector<ACTNormalizedTube> >& all_gt_tubes,
      double* conf_pred_data, double* conf_gt_data);


void ACTEncodeTube(
    const ACTNormalizedTube& prior_tube, const vector<float>& prior_variance,
    const CodeType code_type, const bool encode_variance_in_target,
    const ACTNormalizedTube& tube, ACTNormalizedTube* encode_tube) {
  if (code_type == PriorBoxParameter_CodeType_CORNER) {
    CHECK(false) << "ACT-detector: Not implemented";    
  } else if (code_type == PriorBoxParameter_CodeType_CENTER_SIZE) {
    //ACT-detector: normzalize the difference of center positions and of the sizes independently for each bounding box of the tube

    CHECK_EQ( prior_tube.size(), tube.size()) << "ACT-detector: Tube and PriorTube do have not the same length";
    int sequence_length = prior_tube.size();

    encode_tube->resize(sequence_length);

    //ACT-detector: for each bounding box in the sequence
    for( int jj=0 ; jj<sequence_length ; ++jj){
      float prior_width = prior_tube[jj].xmax() - prior_tube[jj].xmin();
      CHECK_GT(prior_width, 0);
      float prior_height = prior_tube[jj].ymax() - prior_tube[jj].ymin();
      CHECK_GT(prior_height, 0);
      float prior_center_x = (prior_tube[jj].xmin() + prior_tube[jj].xmax()) / 2.;
      float prior_center_y = (prior_tube[jj].ymin() + prior_tube[jj].ymax()) / 2.;

      float bbox_width = tube[jj].xmax() - tube[jj].xmin();
      CHECK_GT(bbox_width, 0);
      float bbox_height = tube[jj].ymax() - tube[jj].ymin();
      CHECK_GT(bbox_height, 0);
      float bbox_center_x = (tube[jj].xmin() + tube[jj].xmax()) / 2.;
      float bbox_center_y = (tube[jj].ymin() + tube[jj].ymax()) / 2.;

      if (encode_variance_in_target) {
        CHECK(false) << "ACT-detector: Not implemented";
      } else {
        // Encode variance in bbox.
        (*encode_tube)[jj].set_xmin(
            (bbox_center_x - prior_center_x) / prior_width / prior_variance[0]);
        (*encode_tube)[jj].set_ymin(
            (bbox_center_y - prior_center_y) / prior_height / prior_variance[1]);
        (*encode_tube)[jj].set_xmax(
            log(bbox_width / prior_width) / prior_variance[2]);
        (*encode_tube)[jj].set_ymax(
            log(bbox_height / prior_height) / prior_variance[3]);
      }

    }
  } else if (code_type == PriorBoxParameter_CodeType_CORNER_SIZE) {
    CHECK(false) << "ACT-detector: Not implemented";
  } else {
    LOG(FATAL) << "Unknown LocLossType.";
  }
}

template <typename Dtype>
void ACTEncodeLocPrediction(const vector<ACTLabelTube>& all_loc_preds,
      const map<int, vector<ACTNormalizedTube> >& all_gt_tubes,
      const vector<map<int, vector<int> > >& all_match_indices,
      const vector<ACTNormalizedTube>& prior_tubes,
      const vector<vector<float> >& prior_variances,
      const MultiBoxLossParameter& multibox_loss_param,
      Dtype* loc_pred_data, Dtype* loc_gt_data) {
  int num = all_loc_preds.size();
  // CHECK_EQ(num, all_match_indices.size());
  // Get parameters.
  const CodeType code_type = multibox_loss_param.code_type();
  const bool encode_variance_in_target =
      multibox_loss_param.encode_variance_in_target();
  const bool bp_inside = multibox_loss_param.bp_inside();  
  int count = 0;
  
  for (int i = 0; i < num; ++i) {
    for (map<int, vector<int> >::const_iterator
         it = all_match_indices[i].begin();
         it != all_match_indices[i].end(); ++it) {
      const int label = it->first;
      const vector<int>& match_index = it->second;
      CHECK(all_loc_preds[i].find(label) != all_loc_preds[i].end());
      const vector<ACTNormalizedTube>& loc_pred =
          all_loc_preds[i].find(label)->second;
      for (int j = 0; j < match_index.size(); ++j) {      
        if (match_index[j] <= -1) {
          continue;
        }
        // Store encoded ground truth.
        const int gt_idx = match_index[j];
        CHECK(all_gt_tubes.find(i) != all_gt_tubes.end());
        CHECK_LT(gt_idx, all_gt_tubes.find(i)->second.size());
        const ACTNormalizedTube& gt_tube = all_gt_tubes.find(i)->second[gt_idx];
        ACTNormalizedTube gt_encode;
        CHECK_LT(j, prior_tubes.size());
        ACTEncodeTube(prior_tubes[j], prior_variances[j], code_type,
                   encode_variance_in_target, gt_tube, &gt_encode);                   
        //ACT-detector: for the whole sequence
        int sequence_length = gt_encode.size();        
        for( int jj = 0; jj<sequence_length ; ++jj){
          loc_gt_data[(count*sequence_length+jj) * 4] = gt_encode[jj].xmin();
          loc_gt_data[(count*sequence_length+jj) * 4 + 1] = gt_encode[jj].ymin();
          loc_gt_data[(count*sequence_length+jj) * 4 + 2] = gt_encode[jj].xmax();
          loc_gt_data[(count*sequence_length+jj) * 4 + 3] = gt_encode[jj].ymax();
        }
        // Store location prediction.
        CHECK_LT(j, loc_pred.size());
        if (bp_inside) {
          CHECK(false) << "ACT-detector: Not implemented";          
        } else {
          //ACT-detector: for the whole sequence
          int sequence_length = loc_pred[j].size();
          for( int jj = 0; jj<sequence_length ; ++jj){
            loc_pred_data[(count*sequence_length+jj) * 4] = loc_pred[j][jj].xmin();
            loc_pred_data[(count*sequence_length+jj) * 4 + 1] = loc_pred[j][jj].ymin();
            loc_pred_data[(count*sequence_length+jj) * 4 + 2] = loc_pred[j][jj].xmax();
            loc_pred_data[(count*sequence_length+jj) * 4 + 3] = loc_pred[j][jj].ymax();
          }
        }
        if (encode_variance_in_target) {
          CHECK(false) << "ACT-detector: Not implemented";          
        }
        ++count;
      }
    }
  }
}

// Explicit initialization.
template void ACTEncodeLocPrediction(const vector<ACTLabelTube>& all_loc_preds,
      const map<int, vector<ACTNormalizedTube> >& all_gt_tubes,
      const vector<map<int, vector<int> > >& all_match_indices,
      const vector<ACTNormalizedTube>& prior_tubes,
      const vector<vector<float> >& prior_variances,
      const MultiBoxLossParameter& multibox_loss_param,
      float* loc_pred_data, float* loc_gt_data);
template void ACTEncodeLocPrediction(const vector<ACTLabelTube>& all_loc_preds,
      const map<int, vector<ACTNormalizedTube> >& all_gt_tubes,
      const vector<map<int, vector<int> > >& all_match_indices,
      const vector<ACTNormalizedTube>& prior_tubes,
      const vector<vector<float> >& prior_variances,
      const MultiBoxLossParameter& multibox_loss_param,
      double* loc_pred_data, double* loc_gt_data);



void ACTClipTube(const ACTNormalizedTube& tube, ACTNormalizedTube* clip_tube) {
  CHECK_EQ(tube.size(),clip_tube->size() ) << "ACT-detector: Tubes do not have the same length";
  //ACT-detector: for each box in the sequence of frames
  for (int jj=0; jj < tube.size(); ++jj){ 
      (*clip_tube)[jj].set_xmin(std::max(std::min(tube[jj].xmin(), 1.f), 0.f));
      (*clip_tube)[jj].set_ymin(std::max(std::min(tube[jj].ymin(), 1.f), 0.f));
      (*clip_tube)[jj].set_xmax(std::max(std::min(tube[jj].xmax(), 1.f), 0.f));
      (*clip_tube)[jj].set_ymax(std::max(std::min(tube[jj].ymax(), 1.f), 0.f));
      (*clip_tube)[jj].clear_size();
      (*clip_tube)[jj].set_size(BBoxSize((*clip_tube)[jj]));
      (*clip_tube)[jj].set_difficult(tube[jj].difficult());
 } 
}

void ACTScaleTube(const ACTNormalizedTube& tube, const int height, const int width,
               ACTNormalizedTube* scale_tube) {
  CHECK_EQ(tube.size(),scale_tube->size() ) << "ACT-detector: Tubes do not have the same length";
  //ACT-detector: for each box in the sequence of frames
  for (int jj=0; jj < tube.size(); ++jj){
      (*scale_tube)[jj].set_xmin(tube[jj].xmin() * width);
      (*scale_tube)[jj].set_ymin(tube[jj].ymin() * height);
      (*scale_tube)[jj].set_xmax(tube[jj].xmax() * width);
      (*scale_tube)[jj].set_ymax(tube[jj].ymax() * height);
      (*scale_tube)[jj].clear_size();
      bool normalized = !(width > 1 || height > 1);
      (*scale_tube)[jj].set_size(BBoxSize((*scale_tube)[jj], normalized));
      (*scale_tube)[jj].set_difficult(tube[jj].difficult());
  }
}

void ACTOutputTube(const ACTNormalizedTube& tube, const pair<int, int>& img_size,
                const bool has_resize, const ResizeParameter& resize_param,
                ACTNormalizedTube* out_tube) {
  const int height = img_size.first;
  const int width = img_size.second;
  ACTNormalizedTube temp_tube = tube;
  if (has_resize && resize_param.resize_mode()) {
    float resize_height = resize_param.height();
    CHECK_GT(resize_height, 0);
    float resize_width = resize_param.width();
    CHECK_GT(resize_width, 0);

    switch (resize_param.resize_mode()) {
      case ResizeParameter_Resize_mode_WARP:
        ACTClipTube(temp_tube, &temp_tube); 
        ACTScaleTube(temp_tube, height, width, out_tube);
        break;
      case ResizeParameter_Resize_mode_FIT_LARGE_SIZE_AND_PAD:
        CHECK(false) << "ACT-detector: Not implemented"; 
        break;
      case ResizeParameter_Resize_mode_FIT_SMALL_SIZE:
        CHECK(false) << "ACT-detector: Not implemented"; 
        break;
      default:
        LOG(FATAL) << "Unknown resize mode.";
    }
  } else {
    // ACT-detector: first clip the normalized tube
    ACTClipTube(temp_tube, &temp_tube);
    // ACT-detector: scale the tube according to the original frame size
    ACTScaleTube(temp_tube, height, width, out_tube);
  }
}

void ACTApplyNMSTubeFast(const vector<ACTNormalizedTube>& tubes,
      const vector<float>& scores, const float score_threshold,
      const float nms_threshold, const float eta, const int top_k,
      vector<int>* indices) {
  // ACT-detector: sanity check
  CHECK_EQ(tubes.size(), scores.size())
      << "ACT-detector: Tubes and scores have different size";

  // Get top_k scores (with corresponding indices).
  vector<pair<float, int> > score_index_vec;
  GetMaxScoreIndex(scores, score_threshold, top_k, &score_index_vec); 

  // Do nms.
  float adaptive_threshold = nms_threshold;
  indices->clear();
  while (score_index_vec.size() != 0) {
    const int idx = score_index_vec.front().second;
    bool keep = true;
    for (int k = 0; k < indices->size(); ++k) {
      if (keep) {
        const int kept_idx = (*indices)[k];
        float overlap = JaccardOverlapTube(tubes[idx], tubes[kept_idx]);
        keep = overlap <= adaptive_threshold;
      } else {
        break;
      }
    }
    if (keep) {
      indices->push_back(idx);
    }
    score_index_vec.erase(score_index_vec.begin());
    if (keep && eta < 1 && adaptive_threshold > 0.5) {
      adaptive_threshold *= eta;
    }
  }
}

template <typename Dtype>
void ACTApplyNMSTubeFast(const Dtype* tubes, const Dtype* scores, const int num,
      const float score_threshold, const float nms_threshold,
      const float eta, const int top_k, vector<int>* indices, 
      const int sequence_length) { 
  // Get top_k scores (with corresponding indices).
  vector<pair<Dtype, int> > score_index_vec;
  GetMaxScoreIndex(scores, num, score_threshold, top_k, &score_index_vec);

  // Do nms.
  float adaptive_threshold = nms_threshold;
  indices->clear();
  while (score_index_vec.size() != 0) {
    const int idx = score_index_vec.front().second;
    bool keep = true;
    for (int k = 0; k < indices->size(); ++k) {
      if (keep) {
        const int kept_idx = (*indices)[k];
        // ACT-detector: multiply by sequence_length here + need to implement another jaccard 
        float overlap = JaccardOverlapTube(tubes + idx * 4 * sequence_length, tubes + kept_idx * 4 * sequence_length, sequence_length); 
        keep = overlap <= adaptive_threshold;
      } else {
        break;
      }
    }
    if (keep) {
      indices->push_back(idx);
    }
    score_index_vec.erase(score_index_vec.begin());
    if (keep && eta < 1 && adaptive_threshold > 0.5) {
      adaptive_threshold *= eta;
    }
  }
}

template
void ACTApplyNMSTubeFast(const float* tubes, const float* scores, const int num,
      const float score_threshold, const float nms_threshold,
      const float eta, const int top_k, vector<int>* indices, 
      const int sequence_length);
template
void ACTApplyNMSTubeFast(const double* tubes, const double* scores, const int num,
      const float score_threshold, const float nms_threshold,
      const float eta, const int top_k, vector<int>* indices, 
      const int sequence_length);

void ACTDecodeTube(
    const ACTNormalizedTube& prior_tube, const vector<float>& prior_variance,
    const CodeType code_type, const bool variance_encoded_in_target,
    const bool clip_bbox, const ACTNormalizedTube& tube,
    ACTNormalizedTube* decode_tube) {
  if (code_type == PriorBoxParameter_CodeType_CORNER) {
    CHECK(false) << "ACT-detector: Not implemented";
  } else if (code_type == PriorBoxParameter_CodeType_CENTER_SIZE) {
    CHECK_EQ( prior_tube.size(), tube.size()) << "ACT-detector: Tube and PriorTube do have not the same length";
    // ACT-detector: for all bounding boxes of the sequence of frames
    int sequence_length = prior_tube.size();
    decode_tube->resize(sequence_length);
    for( int jj=0 ; jj<sequence_length ; ++jj){
        float prior_width = prior_tube[jj].xmax() - prior_tube[jj].xmin();
        CHECK_GT(prior_width, 0);
        float prior_height = prior_tube[jj].ymax() - prior_tube[jj].ymin();
        CHECK_GT(prior_height, 0);
        float prior_center_x = (prior_tube[jj].xmin() + prior_tube[jj].xmax()) / 2.;
        float prior_center_y = (prior_tube[jj].ymin() + prior_tube[jj].ymax()) / 2.;

        float decode_bbox_center_x, decode_bbox_center_y;
        float decode_bbox_width, decode_bbox_height;
        if (variance_encoded_in_target) {
           CHECK(false) << "ACT-detector: Not implemented"; 
        } else {         
          // variance is encoded in bbox, we need to scale the offset accordingly.
          decode_bbox_center_x =
              prior_variance[0] * tube[jj].xmin() * prior_width + prior_center_x;
          decode_bbox_center_y =
              prior_variance[1] * tube[jj].ymin() * prior_height + prior_center_y;
          decode_bbox_width =
              exp(prior_variance[2] * tube[jj].xmax()) * prior_width;
          decode_bbox_height =
              exp(prior_variance[3] * tube[jj].ymax()) * prior_height; 
        }

        (*decode_tube)[jj].set_xmin(decode_bbox_center_x - decode_bbox_width / 2.);
        (*decode_tube)[jj].set_ymin(decode_bbox_center_y - decode_bbox_height / 2.);
        (*decode_tube)[jj].set_xmax(decode_bbox_center_x + decode_bbox_width / 2.);
        (*decode_tube)[jj].set_ymax(decode_bbox_center_y + decode_bbox_height / 2.);
    }
  } else if (code_type == PriorBoxParameter_CodeType_CORNER_SIZE) {
    CHECK(false) << "ACT-detector: Not implemented"; 
  } else {
    LOG(FATAL) << "Unknown LocLossType.";
  }
  // ACT-detector: for each bounding box in the sequence of frames
  for( int jj=0 ; jj < decode_tube->size() ; ++jj){
      float bbox_size = BBoxSize((*decode_tube)[jj]);
      (*decode_tube)[jj].set_size(bbox_size);
      if (clip_bbox) {
        CHECK(false) << "ACT-detector: Not implemented"; 
      }
  }
}

void ACTDecodeTubes(
    const vector<ACTNormalizedTube>& prior_tubes,
    const vector<vector<float> >& prior_variances,
    const CodeType code_type, const bool variance_encoded_in_target,
    const bool clip_bbox, const vector<ACTNormalizedTube>& tubes,
    vector<ACTNormalizedTube>* decode_tubes) {
  CHECK_EQ(prior_tubes.size(), prior_variances.size());
  CHECK_EQ(prior_tubes.size(), tubes.size());
  int num_tubes = prior_tubes.size();
  if (num_tubes >= 1) {
    CHECK_EQ(prior_variances[0].size(), 4);
  }
  decode_tubes->clear();
  for (int i = 0; i < num_tubes; ++i) {
    ACTNormalizedTube decode_tube;
    ACTDecodeTube(prior_tubes[i], prior_variances[i], code_type,
               variance_encoded_in_target, clip_bbox, tubes[i], &decode_tube);
    decode_tubes->push_back(decode_tube);
  }
}

void ACTDecodeTubesAll(const vector<ACTLabelTube>& all_loc_preds,
    const vector<ACTNormalizedTube>& prior_tubes,
    const vector<vector<float> >& prior_variances,
    const int num, const bool share_location,
    const int num_loc_classes, const int background_label_id,
    const CodeType code_type, const bool variance_encoded_in_target,
    const bool clip, vector<ACTLabelTube>* all_decode_tubes) {
  CHECK_EQ(all_loc_preds.size(), num);
  all_decode_tubes->clear();
  all_decode_tubes->resize(num);
  for (int i = 0; i < num; ++i) {
    // Decode predictions into bboxes.
    ACTLabelTube& decode_tubes = (*all_decode_tubes)[i];
    for (int c = 0; c < num_loc_classes; ++c) {
      int label = share_location ? -1 : c;
      if (label == background_label_id) {
        // Ignore background class.
        continue;
      }
      if (all_loc_preds[i].find(label) == all_loc_preds[i].end()) {
        // Something bad happened if there are no predictions for current label.
        LOG(FATAL) << "Could not find location predictions for label " << label;
      }
      const vector<ACTNormalizedTube>& label_loc_preds =
          all_loc_preds[i].find(label)->second;
      ACTDecodeTubes(prior_tubes, prior_variances,
                     code_type, variance_encoded_in_target, clip,
                     label_loc_preds, &(decode_tubes[label]));  
    }
  }
}


}  // namespace caffe
