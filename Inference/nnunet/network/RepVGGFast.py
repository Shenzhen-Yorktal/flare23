import torch
import numpy as np

from torch import nn
from copy import deepcopy
from nnunet.network.neural_network import SegmentationNetwork
from nnunet.utilities.utilities_stuff import Upsample


class RepVGGLayerInference(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, props, stride=None):
        super().__init__()

        self.kernel_size = kernel_size
        props['conv_op_kwargs']['stride'] = 1

        self.stride = stride
        self.props = props
        self.out_planes = out_planes
        self.in_planes = in_planes

        if stride is not None:
            kwargs_conv1 = deepcopy(props['conv_op_kwargs'])
            kwargs_conv1['stride'] = stride
        else:
            kwargs_conv1 = props['conv_op_kwargs']

        self.conv1 = props['conv_op'](in_planes, out_planes, kernel_size, padding=[(i - 1) // 2 for i in kernel_size],
                                      **kwargs_conv1)

        self.nonlin = props['nonlin'](**props['nonlin_kwargs'])

    def forward(self, x):
        out = self.conv1(x)
        return self.nonlin(out)


class RepVGGBInferenceBlock(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, network_props, num_blocks, first_stride=None):
        super().__init__()

        network_props = deepcopy(network_props)  # network_props is a dict and mutable, so we deepcopy to be safe.

        self.convs = nn.Sequential(
            RepVGGLayerInference(input_channels, output_channels, kernel_size, network_props, first_stride),
            *[RepVGGLayerInference(output_channels, output_channels, kernel_size, network_props) for _ in
              range(2 * num_blocks - 1)]
        )

    def forward(self, x):
        return self.convs(x)


class RepVGGUNetEncoder(nn.Module):
    def __init__(self, input_channels, base_num_features, num_blocks_per_stage, feat_map_mul_on_downscale,
                 pool_op_kernel_sizes, conv_kernel_sizes, props, default_return_skips=True,
                 max_num_features=480):
        super(RepVGGUNetEncoder, self).__init__()

        self.default_return_skips = default_return_skips
        self.props = props

        self.stages = []
        self.stage_output_features = []
        self.stage_pool_kernel_size = []
        self.stage_conv_op_kernel_size = []

        num_stages = len(conv_kernel_sizes)

        if not isinstance(num_blocks_per_stage, (list, tuple)):
            num_blocks_per_stage = [num_blocks_per_stage] * num_stages

        self.num_blocks_per_stage = num_blocks_per_stage

        self.initial_conv = props['conv_op'](input_channels, base_num_features, 3, padding=1, **props['conv_op_kwargs'])
        self.initial_nonlin = props['nonlin'](**props['nonlin_kwargs'])

        current_input_features = base_num_features
        for stage in range(num_stages):
            current_output_features = min(base_num_features * feat_map_mul_on_downscale ** stage, max_num_features)
            current_kernel_size = conv_kernel_sizes[stage]
            current_pool_kernel_size = pool_op_kernel_sizes[stage]

            current_stage = RepVGGBInferenceBlock(current_input_features, current_output_features, current_kernel_size,
                                                  props,
                                                  self.num_blocks_per_stage[stage], current_pool_kernel_size)

            self.stages.append(current_stage)
            self.stage_output_features.append(current_output_features)
            self.stage_conv_op_kernel_size.append(current_kernel_size)
            self.stage_pool_kernel_size.append(current_pool_kernel_size)

            current_input_features = current_output_features

        self.stages = nn.ModuleList(self.stages)

    def forward(self, x, return_skips=None):
        skips = []

        x = self.initial_nonlin(self.initial_conv(x))
        for s in self.stages:
            x = s(x)
            if self.default_return_skips:
                skips.append(x)

        if return_skips is None:
            return_skips = self.default_return_skips

        if return_skips:
            return skips
        else:
            return x


class RepVGGUNetDecoder(nn.Module):
    def __init__(self, previous, num_classes, num_blocks_per_stage=None, network_props=None, deep_supervision=False,
                 upscale_logits=False):
        super(RepVGGUNetDecoder, self).__init__()
        self.num_classes = num_classes
        self.deep_supervision = deep_supervision
        previous_stages = previous.stages
        previous_stage_output_features = previous.stage_output_features
        previous_stage_pool_kernel_size = previous.stage_pool_kernel_size
        previous_stage_conv_op_kernel_size = previous.stage_conv_op_kernel_size

        if network_props is None:
            self.props = previous.props
        else:
            self.props = network_props

        if self.props['conv_op'] == nn.Conv2d:
            transpconv = nn.ConvTranspose2d
            upsample_mode = "bilinear"
        elif self.props['conv_op'] == nn.Conv3d:
            transpconv = nn.ConvTranspose3d
            upsample_mode = "trilinear"
        else:
            raise ValueError("unknown convolution dimensionality, conv op: %s" % str(self.props['conv_op']))

        if num_blocks_per_stage is None:
            num_blocks_per_stage = previous.num_blocks_per_stage[:-1][::-1]

        self.stage_pool_kernel_size = previous_stage_pool_kernel_size
        self.stage_output_features = previous_stage_output_features
        self.stage_conv_op_kernel_size = previous_stage_conv_op_kernel_size

        num_stages = len(previous_stages) - 1

        self.tus = []
        self.stages = []
        self.deep_supervision_outputs = []

        cum_upsample = np.cumprod(np.vstack(self.stage_pool_kernel_size), axis=0).astype(int)
        features_skip = None
        for i, s in enumerate(np.arange(num_stages)[::-1]):
            features_below = previous_stage_output_features[s + 1]
            features_skip = previous_stage_output_features[s]

            self.tus.append(transpconv(features_below, features_skip, previous_stage_pool_kernel_size[s + 1],
                                       previous_stage_pool_kernel_size[s + 1], bias=False))

            self.stages.append(
                RepVGGBInferenceBlock(2 * features_skip, features_skip, previous_stage_conv_op_kernel_size[s],
                                      self.props, num_blocks_per_stage[i], None))

            if deep_supervision and s != 0:
                seg_layer = self.props['conv_op'](features_skip, num_classes, 1, 1, 0, 1, 1, False)
                if upscale_logits:
                    upsample = Upsample(scale_factor=cum_upsample[s], mode=upsample_mode)
                    self.deep_supervision_outputs.append(nn.Sequential(seg_layer, upsample))
                else:
                    self.deep_supervision_outputs.append(seg_layer)

        self.segmentation_output = self.props['conv_op'](features_skip, num_classes, 1, 1, 0, 1, 1, False)

        self.tus = nn.ModuleList(self.tus)
        self.stages = nn.ModuleList(self.stages)
        self.deep_supervision_outputs = nn.ModuleList(self.deep_supervision_outputs)

    def forward(self, skips):
        skips = skips[::-1]
        seg_outputs = []

        x = skips[0]

        for i in range(len(self.tus)):
            x = self.tus[i](x)
            x = torch.cat((x, skips[i + 1]), dim=1)
            x = self.stages[i](x)

        segmentation = self.segmentation_output(x)

        return segmentation


class VGGFastUNet(SegmentationNetwork):
    def __init__(self, input_channels, base_num_features, num_blocks_per_stage_encoder, feat_map_mul_on_downscale,
                 pool_op_kernel_sizes, conv_kernel_sizes, props, num_classes, num_blocks_per_stage_decoder,
                 deep_supervision=False, upscale_logits=False, max_features=512, initializer=None, props_decoder=None):
        super().__init__()
        self.conv_op = props['conv_op']
        self.num_classes = num_classes

        self.encoder = RepVGGUNetEncoder(input_channels, base_num_features, num_blocks_per_stage_encoder,
                                         feat_map_mul_on_downscale, pool_op_kernel_sizes, conv_kernel_sizes,
                                         props, default_return_skips=True, max_num_features=max_features)
        props['dropout_op_kwargs']['p'] = 0
        if props_decoder is None:
            props_decoder = props
        self.decoder = RepVGGUNetDecoder(self.encoder, num_classes, num_blocks_per_stage_decoder, props_decoder,
                                         deep_supervision, upscale_logits)
        if initializer is not None:
            self.apply(initializer)

    def forward(self, x):
        skips = self.encoder(x)
        return self.decoder(skips)

    def _internal_predict_3D_3Dconv(self, x, step_size, patch_size):
        step = [16, 16, 16]
        patch_size = [((i - 1) // j + 1) * j for i, j in zip(x.shape[1:], step)]
        data, slicer = self._pad_nd_image(x, patch_size)
        data = torch.from_numpy(data).cuda(self.get_device(), non_blocking=True)
        predicted_probabilities = self._internal_maybe_mirror_and_pred_3D(data[None], None)[0]

        slicer = tuple(
            [slice(0, predicted_probabilities.shape[i]) for i in range(len(predicted_probabilities.shape) -
                                                                       (len(slicer) - 1))] + slicer[1:])
        predicted_probabilities = predicted_probabilities[slicer]

        predicted_segmentation = predicted_probabilities.argmax(0)
        predicted_segmentation = predicted_segmentation.detach().cpu().numpy()

        return predicted_segmentation
        
    def _internal_predict_3D_3Dconv_tiled(self, x, step_size, patch_size):
        data, slicer = self._pad_nd_image(x, patch_size)
        data_shape = data.shape

        steps = self._compute_steps_for_sliding_window(patch_size, data_shape[1:], step_size)
        num_tiles = len(steps[0]) * len(steps[1]) * len(steps[2])

        temp_size = [k for k in patch_size]
        add_for_nb_of_preds = torch.ones(temp_size, device=self.get_device())
        aggregated_results = torch.zeros([self.num_classes] + list(data.shape[1:]), dtype=torch.half,
                                         device=self.get_device())

        data = torch.from_numpy(data).cuda(self.get_device(), non_blocking=True)
        aggregated_nb_of_predictions = torch.zeros(list(data.shape[1:]), dtype=torch.half,
                                                   device=self.get_device())

        for x in steps[0]:
            lb_x = x
            ub_x = x + patch_size[0]
            for y in steps[1]:
                lb_y = y
                ub_y = y + patch_size[1]
                for z in steps[2]:
                    lb_z = z
                    ub_z = z + patch_size[2]

                    predicted_patch = self._internal_maybe_mirror_and_pred_3D(
                        data[None, :, lb_x:ub_x, lb_y:ub_y, lb_z:ub_z], None)[0]

                    predicted_patch = predicted_patch.half()

                    aggregated_results[:, lb_x:ub_x, lb_y:ub_y, lb_z:ub_z] += predicted_patch
                    aggregated_nb_of_predictions[lb_x:ub_x, lb_y:ub_y, lb_z:ub_z] += add_for_nb_of_preds

        # we reverse the padding here (remeber that we padded the input to be at least as large as the patch size
        slicer = tuple(
            [slice(0, aggregated_results.shape[i]) for i in
             range(len(aggregated_results.shape) - (len(slicer) - 1))] + slicer[1:])
        aggregated_results = aggregated_results[slicer]
        aggregated_nb_of_predictions = aggregated_nb_of_predictions[slicer[1:]]

        # computing the class_probabilities by dividing the aggregated result with result_numsamples
        for temp_c in range(aggregated_results.shape[0]):
            aggregated_results[temp_c] /= aggregated_nb_of_predictions

        predicted_segmentation = aggregated_results.argmax(0)
        return predicted_segmentation.detach().cpu().numpy()
