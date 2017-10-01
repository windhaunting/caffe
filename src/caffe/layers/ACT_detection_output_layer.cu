#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#endif  // USE_OPENCV
#include <algorithm>
#include <fstream>  // NOLINT(readability/streams)
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "boost/filesystem.hpp"
#include "boost/foreach.hpp"

#include "caffe/layers/ACT_detection_output_layer.hpp"

namespace caffe {

template <typename Dtype>
void ACTDetectionOutputLayer<Dtype>::Forward_gpu( 
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const Dtype* loc_data = bottom[0]->gpu_data();
  const Dtype* prior_data = bottom[2]->gpu_data();
  const int num = bottom[0]->num();

  // Decode predictions.
  Dtype* bbox_data = bbox_preds_.mutable_gpu_data();
  const int loc_count = bbox_preds_.count();
  const bool clip_bbox = false;
  ACTDecodeTubesGPU<Dtype>(loc_count, loc_data, prior_data, code_type_,
      variance_encoded_in_target_, num_priors_, share_location_,
      num_loc_classes_, background_label_id_, clip_bbox, bbox_data, 
      sequence_length_); 
  // Retrieve all decoded location predictions.
  const Dtype* bbox_cpu_data;
  if (!share_location_) {
    Dtype* bbox_permute_data = bbox_permute_.mutable_gpu_data();
    // ACT-detector: keep whole tubes coordinates: 4*sequence_length 
    PermuteDataGPU<Dtype>(loc_count, bbox_data, num_loc_classes_, num_priors_,
        4*sequence_length_, bbox_permute_data); 
  } else {
    bbox_cpu_data = bbox_preds_.cpu_data();
  }

  // Retrieve all confidences.
  Dtype* conf_permute_data = conf_permute_.mutable_gpu_data();
  // ACT-detector: do not change the confidence score
  PermuteDataGPU<Dtype>(bottom[1]->count(), bottom[1]->gpu_data(),
      num_classes_, num_priors_, 1, conf_permute_data);  
  const Dtype* conf_cpu_data = conf_permute_.cpu_data();

  // ACT-detector: save all tubes/scores
  if( save_full_){
    vector<int> top_shape(2, 1);
    // ACT-detector: num_priors are num_prior tubes
    top_shape.push_back(num_priors_ * num ); 
    // ACT-detector: first is the index of the image in the batch, then tube coordinates, then class scores
    int ncols = 1 + 4* sequence_length_ + num_classes_; 
    top_shape.push_back(ncols); 
    top[0]->Reshape(top_shape);
    Dtype* top_data = top[0]->mutable_cpu_data();
    
    // ACT-detector: for each image in the batch
    for (int i = 0; i < num; ++i) { 
      const int conf_idx = i * num_classes_ * num_priors_;
      CHECK(share_location_) << "ACT_detection_output_layer: save full is implemented only with share location";

      // ACT-detector: only if share location
      const int bbox_idx = i * num_priors_ * 4 * sequence_length_; 
      const Dtype* cur_bbox_data = bbox_cpu_data + bbox_idx;
      // ACT-detector: scores are stored consecutively for each class 
      const Dtype* cur_conf_data = conf_cpu_data + conf_idx; 
      
      for(int j=0 ; j<num_priors_ ; ++j){
        const int start_idx = i*num_priors_*ncols + j*ncols;
        // ACT-detector: batch idx
        top_data[ start_idx    ] = i;
        // ACT-detector: for each box in the sequence
        for (int jj=0; jj< sequence_length_ ; ++jj){
            for (int k = 0; k < 4; ++k) {
              // ACT-detector: replace this double for loop with a much faster caffe_copy?
              top_data[start_idx + 4 * jj + 1 + k] = cur_bbox_data[j * 4 * sequence_length_ + 4*jj+ k]; 
            }
        }
        // ACT-detector: scoring
        for(int c=0 ; c<num_classes_ ; ++c){
          top_data[start_idx + 1 + 4*sequence_length_ + c] = cur_conf_data[c*num_priors_+j];
        }
      }
      
    }
    return; // ACT-detector: finished saving all tubes/scores
  }
  
  
  int num_kept = 0;
  vector<map<int, vector<int> > > all_indices;
  for (int i = 0; i < num; ++i) {
    map<int, vector<int> > indices;
    int num_det = 0;
    // ACT-detector: keep one confidence per tube
    const int conf_idx = i * num_classes_ * num_priors_; 
    // ACT-detector: starting index of the tubes in the loc_data    
    int bbox_idx; 
    if (share_location_) {
      // ACT-detector: process the boxes of the whole sequence
      bbox_idx = i * num_priors_ * 4 * sequence_length_;
    } else {
      bbox_idx = conf_idx * 4 * sequence_length_;
    }
    for (int c = 0; c < num_classes_; ++c) {
      if (c == background_label_id_) {
        // Ignore background class.
        continue;
      }
      // ACT-detector: keep one confidence per tube
      const Dtype* cur_conf_data = conf_cpu_data + conf_idx + c * num_priors_; 
      // ACT-detector: reminder: bbox_idx is starting index of the tubes!   
      const Dtype* cur_bbox_data = bbox_cpu_data + bbox_idx;
      if (!share_location_) {
        // ACT-detector: parse the box coordinates for the sequence 
        cur_bbox_data += c * num_priors_ * 4 * sequence_length_;
      }
      

      ACTApplyNMSTubeFast(cur_bbox_data, cur_conf_data, num_priors_,
          confidence_threshold_, nms_threshold_, eta_, top_k_, &(indices[c]), 
          sequence_length_);           
      num_det += indices[c].size();
    }     
    if (keep_top_k_ > -1 && num_det > keep_top_k_) {
      vector<pair<float, pair<int, int> > > score_index_pairs;
      for (map<int, vector<int> >::iterator it = indices.begin();
           it != indices.end(); ++it) {
        int label = it->first;
        const vector<int>& label_indices = it->second;
        for (int j = 0; j < label_indices.size(); ++j) {
          int idx = label_indices[j];
          float score = conf_cpu_data[conf_idx + label * num_priors_ + idx];
          score_index_pairs.push_back(std::make_pair(
                  score, std::make_pair(label, idx)));
        }
      }
      // Keep top k results per image.
      std::sort(score_index_pairs.begin(), score_index_pairs.end(),
                SortScorePairDescend<pair<int, int> >);
      score_index_pairs.resize(keep_top_k_);
      // Store the new indices.
      map<int, vector<int> > new_indices;
      for (int j = 0; j < score_index_pairs.size(); ++j) {
        int label = score_index_pairs[j].second.first;
        int idx = score_index_pairs[j].second.second;
        new_indices[label].push_back(idx);
      }
      all_indices.push_back(new_indices);
      num_kept += keep_top_k_;
    } else {
      all_indices.push_back(indices);
      num_kept += num_det;
    }
  }

  vector<int> top_shape(2, 1);
  top_shape.push_back(num_kept);
  // ACT-detector: for the whole sequence! 
  top_shape.push_back(3 + 4* sequence_length_);
  Dtype* top_data;
  if (num_kept == 0) {
    LOG(INFO) << "Couldn't find any detections";
    top_shape[2] = num;
    top[0]->Reshape(top_shape);
    top_data = top[0]->mutable_cpu_data();
    caffe_set<Dtype>(top[0]->count(), -1, top_data);
    // Generate fake results per image.
    for (int i = 0; i < num; ++i) {
      top_data[0] = i;
      // ACT-detector: dimensionality of each row for the whole sequence
      top_data += 3 + 4* sequence_length_;
    }
  } else {
    top[0]->Reshape(top_shape);
    top_data = top[0]->mutable_cpu_data();
  }

  int count = 0;
  boost::filesystem::path output_directory(output_directory_);
  for (int i = 0; i < num; ++i) {     
    const int conf_idx = i * num_classes_ * num_priors_;
    int bbox_idx;
    // ACT-detector: bbox_idx is the idx of the tube
    if (share_location_) {
      bbox_idx = i * num_priors_ * 4 * sequence_length_;
    } else {
      bbox_idx = conf_idx * 4 * sequence_length_;
    }
    for (map<int, vector<int> >::iterator it = all_indices[i].begin();
         it != all_indices[i].end(); ++it) {
      int label = it->first;
      vector<int>& indices = it->second;
      if (need_save_) {
        CHECK(label_to_name_.find(label) != label_to_name_.end())
          << "Cannot find label: " << label << " in the label map.";
        CHECK_LT(name_count_, names_.size());
      }
      const Dtype* cur_conf_data =
        conf_cpu_data + conf_idx + label * num_priors_;
      const Dtype* cur_bbox_data = bbox_cpu_data + bbox_idx;
      if (!share_location_) {
        // ACT-detector: for the whole sequence
        cur_bbox_data += label * num_priors_ * 4 * sequence_length_;
      } 
      // ACT-detector: each row is a [3 dimensional vector + 4x sequence_length vector]
      // eg. if sequence_length = 6, each row is 27D
      // the 3D vector is: [image_id, label, confidence] 
      // sequence_length x the coordinates: [ xmin, ymin, xmax, ymax] 
      int row_size = 3 + 4* sequence_length_; 
      for (int j = 0; j < indices.size(); ++j) {
        int idx = indices[j];
        top_data[count * row_size] = i;
        top_data[count * row_size + 1] = label;
        top_data[count * row_size + 2] = cur_conf_data[idx]; 
        // ACT-detector: for all bounding boxes in the sequence of frames        
        for (int jj=0; jj< sequence_length_ ; ++jj){
            for (int k = 0; k < 4; ++k) {
              top_data[count * row_size + 4 * jj + 3 + k] = cur_bbox_data[idx * 4 * sequence_length_ + 4*jj+ k];
            }
        }        
        if (need_save_) {
          // Generate output bbox.
          CHECK(false) << "ACT-detector: Saving tubes is not implemeted"; 
        }
        ++count;
      }
    }
    if (need_save_) {
      CHECK(false) << "ACT-detector: Saving tubes is not implemeted";      
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(ACTDetectionOutputLayer);

}  // namespace caffe
