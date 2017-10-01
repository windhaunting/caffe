#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#endif  // USE_OPENCV

#ifndef CAFFE_UTIL_ACT_CUBOID_UTIL_H_
#define CAFFE_UTIL_ACT_CUBOID_UTIL_H_


#include <stdint.h>
#include <cmath>  // for std::fabs and std::signbit
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "glog/logging.h"

#include "caffe/caffe.hpp"

namespace caffe {
    typedef EmitConstraint_EmitType EmitType;
    typedef PriorBoxParameter_CodeType CodeType;
    typedef MultiBoxLossParameter_MatchType MatchType;
    typedef MultiBoxLossParameter_LocLossType LocLossType;
    typedef MultiBoxLossParameter_ConfLossType ConfLossType;
    typedef MultiBoxLossParameter_MiningType MiningType;

    typedef vector<NormalizedBBox> ACTNormalizedTube;
    typedef map<int, vector<ACTNormalizedTube> > ACTLabelTube;

    template <typename Dtype>
    void ACTGetGroundTruth(const Dtype* gt_data, const int num_gt,
          const int background_label_id, const bool use_difficult_gt,
          map<int, vector<ACTNormalizedTube> >* all_gt_tubes, const int sequence_length);


    template <typename Dtype>
    void ACTGetPriorTubes(const Dtype* prior_data, const int num_priors,
          vector<ACTNormalizedTube>* prior_tubes,
          vector<vector<float> >* prior_variances,
          const int sequence_length);

    template <typename Dtype>
    void ACTGetLocPredictions(const Dtype* loc_data, const int num,
          const int num_preds_per_class, const int num_loc_classes,
          const bool share_location, vector<ACTLabelTube>* loc_preds,
          const int sequence_length);

    void ACTFindMatches(const vector<ACTLabelTube>& all_loc_preds,
          const map<int, vector<ACTNormalizedTube> >& all_gt_tubes,
          const vector<ACTNormalizedTube>& prior_tubes,
          const vector<vector<float> >& prior_variances,
          const MultiBoxLossParameter& multibox_loss_param,
          vector<map<int, vector<float> > >* all_match_overlaps,
          vector<map<int, vector<int> > >* all_match_indices);
          
    template <typename Dtype>
      void ACTComputeConfLoss(const Dtype* conf_data, const int num,
          const int num_preds_per_class, const int num_classes,
          const int background_label_id, const ConfLossType loss_type,
          const vector<map<int, vector<int> > >& all_match_indices,
          const map<int, vector<ACTNormalizedTube> >& all_gt_tubes,
          vector<vector<float> >* all_conf_loss);


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
        vector<vector<int> >* all_neg_indices);

    template <typename Dtype>
    void ACTEncodeConfPrediction(const Dtype* conf_data, const int num,
          const int num_priors, const MultiBoxLossParameter& multibox_loss_param,
          const vector<map<int, vector<int> > >& all_match_indices,
          const vector<vector<int> >& all_neg_indices,
          const map<int, vector<ACTNormalizedTube> >& all_gt_tubes,
          Dtype* conf_pred_data, Dtype* conf_gt_data);


    template <typename Dtype>
    void ACTEncodeLocPrediction(const vector<ACTLabelTube>& all_loc_preds,
          const map<int, vector<ACTNormalizedTube> >& all_gt_tubes,
          const vector<map<int, vector<int> > >& all_match_indices,
          const vector<ACTNormalizedTube>& prior_tubes,
          const vector<vector<float> >& prior_variances,
          const MultiBoxLossParameter& multibox_loss_param,
          Dtype* loc_pred_data, Dtype* loc_gt_data);

    void ACTEncodeTube(
        const ACTNormalizedTube& prior_tube, const vector<float>& prior_variance,
        const CodeType code_type, const bool encode_variance_in_target,
        const ACTNormalizedTube& tube, ACTNormalizedTube* encode_tube);

    // Compute the jaccard (intersection over union IoU) overlap between two bboxes.    
    float JaccardOverlapTube(const ACTNormalizedTube& tube1, const ACTNormalizedTube& tube2,
                     const bool normalized = true);

    // test stuff

    void ACTApplyNMSTubeFast(const vector<ACTNormalizedTube>& tubes,
      const vector<float>& scores, const float score_threshold,
      const float nms_threshold, const float eta, const int top_k,
      vector<int>* indices);

    void ACTDecodeTubesAll(const vector<ACTLabelTube>& all_loc_preds,
        const vector<ACTNormalizedTube>& prior_tubes,
        const vector<vector<float> >& prior_variances,
        const int num, const bool share_location,
        const int num_loc_classes, const int background_label_id,
        const CodeType code_type, const bool variance_encoded_in_target,
        const bool clip, vector<ACTLabelTube>* all_decode_tubes);

    void ACTOutputTube(const ACTNormalizedTube& tube, const pair<int, int>& img_size,
                const bool has_resize, const ResizeParameter& resize_param,
                ACTNormalizedTube* out_tube);

#ifndef CPU_ONLY  // GPU
    template <typename Dtype>
      void ACTComputeConfLossGPU(const Blob<Dtype>& conf_blob, const int num,
          const int num_preds_per_class, const int num_classes,
          const int background_label_id, const ConfLossType loss_type,
          const vector<map<int, vector<int> > >& all_match_indices,
          const map<int, vector<ACTNormalizedTube> >& all_gt_tubes,
          vector<vector<float> >* all_conf_loss);
          
    template <typename Dtype>
    void ACTDecodeTubesGPU(const int nthreads,
              const Dtype* loc_data, const Dtype* prior_data,
              const CodeType code_type, const bool variance_encoded_in_target,
              const int num_priors, const bool share_location,
              const int num_loc_classes, const int background_label_id,
              const bool clip_bbox, Dtype* bbox_data, const int sequence_length);
          
#endif // END GPU

    template <typename Dtype>
    void ACTApplyNMSTubeFast(const Dtype* tubes, const Dtype* scores, const int num,
          const float score_threshold, const float nms_threshold,
          const float eta, const int top_k, vector<int>* indices, 
          const int sequence_length); 


          
}

#endif // CAFFE_UTIL_ACT_CUBOID_UTIL_H_
