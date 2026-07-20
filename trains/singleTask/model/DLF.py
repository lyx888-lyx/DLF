"""
here is the mian backbone for DLF
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from ...subNets import BertTextEncoder
from ...subNets.transformers_encoder.transformer import TransformerEncoder   

class DLF(nn.Module):
    def __init__(self, args):
        super(DLF, self).__init__()
        if args.use_bert:
            self.text_model = BertTextEncoder(use_finetune=args.use_finetune, transformers=args.transformers,
                                              pretrained=args.pretrained)
        self.use_bert = args.use_bert
        dst_feature_dims, nheads = args.dst_feature_dim_nheads
        if args.dataset_name == 'mosi':
            if args.need_data_aligned:
                self.len_l, self.len_v, self.len_a = 50, 50, 50
            else:
                self.len_l, self.len_v, self.len_a = 50, 500, 375
        if args.dataset_name == 'mosei':
            if args.need_data_aligned:
                self.len_l, self.len_v, self.len_a = 50, 50, 50
            else:
                self.len_l, self.len_v, self.len_a = 50, 500, 500
        self.orig_d_l, self.orig_d_a, self.orig_d_v = args.feature_dims
        self.d_l = self.d_a = self.d_v = dst_feature_dims
        self.num_heads = nheads     
        self.layers = args.nlevels 
        self.attn_dropout = args.attn_dropout
        self.attn_dropout_a = args.attn_dropout_a
        self.attn_dropout_v = args.attn_dropout_v
        self.relu_dropout = args.relu_dropout
        self.embed_dropout = args.embed_dropout
        self.res_dropout = args.res_dropout
        self.output_dropout = args.output_dropout
        self.text_dropout = args.text_dropout
        self.attn_mask = args.attn_mask
        combined_dim_low = self.d_a    
        combined_dim_high = self.d_a 
        combined_dim = (self.d_l + self.d_a + self.d_v ) + self.d_l * 3  
        
        output_dim = 1

        # 1. Temporal convolutional layers for initial feature
        self.proj_l = nn.Conv1d(self.orig_d_l, self.d_l, kernel_size=args.conv1d_kernel_size_l, padding=0, bias=False)
        self.proj_a = nn.Conv1d(self.orig_d_a, self.d_a, kernel_size=args.conv1d_kernel_size_a, padding=0, bias=False)
        self.proj_v = nn.Conv1d(self.orig_d_v, self.d_v, kernel_size=args.conv1d_kernel_size_v, padding=0, bias=False)

        # 2. Modality-specific encoder
        self.encoder_s_l = self.get_network(self_type='l', layers = self.layers)       
        self.encoder_s_v = self.get_network(self_type='v', layers = self.layers)
        self.encoder_s_a = self.get_network(self_type='a', layers = self.layers)

        #   Modality-shared encoder 
        self.encoder_c = self.get_network(self_type='l', layers = self.layers)        
        

        # 3. Decoder for reconstruct three modalities
        self.decoder_l = nn.Conv1d(self.d_l * 2, self.d_l, kernel_size=1, padding=0, bias=False)     
        self.decoder_v = nn.Conv1d(self.d_v * 2, self.d_v, kernel_size=1, padding=0, bias=False)
        self.decoder_a = nn.Conv1d(self.d_a * 2, self.d_a, kernel_size=1, padding=0, bias=False)

        # for calculate cosine sim between s_x
        self.proj_cosine_l = nn.Linear(combined_dim_low * (self.len_l - args.conv1d_kernel_size_l + 1), combined_dim_low)     
        self.proj_cosine_v = nn.Linear(combined_dim_low * (self.len_v - args.conv1d_kernel_size_v + 1), combined_dim_low)
        self.proj_cosine_a = nn.Linear(combined_dim_low * (self.len_a - args.conv1d_kernel_size_a + 1), combined_dim_low)

        # for align c_l, c_v, c_a
        self.align_c_l = nn.Linear(combined_dim_low * (self.len_l - args.conv1d_kernel_size_l + 1), combined_dim_low)
        self.align_c_v = nn.Linear(combined_dim_low * (self.len_v - args.conv1d_kernel_size_v + 1), combined_dim_low)
        self.align_c_a = nn.Linear(combined_dim_low * (self.len_a - args.conv1d_kernel_size_a + 1), combined_dim_low)

        self.self_attentions_c_l = self.get_network(self_type='l')
        self.self_attentions_c_v = self.get_network(self_type='v')
        self.self_attentions_c_a = self.get_network(self_type='a')

        self.proj1_c = nn.Linear(self.d_l * 3, self.d_l * 3)
        self.proj2_c = nn.Linear(self.d_l * 3, self.d_l * 3)
        self.out_layer_c = nn.Linear(self.d_l * 3, output_dim)


        # 4 Multimodal Crossmodal Attentions
        self.trans_l_with_a = self.get_network(self_type='la', layers = self.layers)  
        self.trans_l_with_v = self.get_network(self_type='lv', layers = self.layers) 
        self.trans_a_with_l = self.get_network(self_type='al')
        self.trans_a_with_v = self.get_network(self_type='av')
        self.trans_v_with_l = self.get_network(self_type='vl')
        self.trans_v_with_a = self.get_network(self_type='va')
        self.trans_l_mem = self.get_network(self_type='l_mem', layers=self.layers)
        self.trans_a_mem = self.get_network(self_type='a_mem', layers=3)
        self.trans_v_mem = self.get_network(self_type='v_mem', layers=3)


        # 5. fc layers for shared features
        self.proj1_l_low = nn.Linear(combined_dim_low * (self.len_l - args.conv1d_kernel_size_l + 1), combined_dim_low)
        self.proj2_l_low = nn.Linear(combined_dim_low, combined_dim_low * (self.len_l - args.conv1d_kernel_size_l + 1))
        self.out_layer_l_low = nn.Linear(combined_dim_low * (self.len_l - args.conv1d_kernel_size_l + 1), output_dim)
        self.proj1_v_low = nn.Linear(combined_dim_low * (self.len_v - args.conv1d_kernel_size_v + 1), combined_dim_low)
        self.proj2_v_low = nn.Linear(combined_dim_low, combined_dim_low * (self.len_v - args.conv1d_kernel_size_v + 1))
        self.out_layer_v_low = nn.Linear(combined_dim_low * (self.len_v - args.conv1d_kernel_size_v + 1), output_dim)
        self.proj1_a_low = nn.Linear(combined_dim_low * (self.len_a - args.conv1d_kernel_size_a + 1), combined_dim_low)
        self.proj2_a_low = nn.Linear(combined_dim_low, combined_dim_low * (self.len_a - args.conv1d_kernel_size_a + 1))
        self.out_layer_a_low = nn.Linear(combined_dim_low * (self.len_a - args.conv1d_kernel_size_a + 1), output_dim)

        
        # 6. fc layers for specific features
        self.proj1_l_high = nn.Linear(combined_dim_high, combined_dim_high)
        self.proj2_l_high = nn.Linear(combined_dim_high, combined_dim_high)
        self.out_layer_l_high = nn.Linear(combined_dim_high, output_dim)
        self.proj1_v_high = nn.Linear(combined_dim_high, combined_dim_high)
        self.proj2_v_high = nn.Linear(combined_dim_high, combined_dim_high)
        self.out_layer_v_high = nn.Linear(combined_dim_high, output_dim)
        self.proj1_a_high = nn.Linear(combined_dim_high, combined_dim_high)
        self.proj2_a_high = nn.Linear(combined_dim_high, combined_dim_high)
        self.out_layer_a_high = nn.Linear(combined_dim_high, output_dim)
        
        # 7. project for fusion
        self.projector_l = nn.Linear(self.d_l, self.d_l)
        self.projector_v = nn.Linear(self.d_v, self.d_v)
        self.projector_a = nn.Linear(self.d_a, self.d_a)
        self.projector_c = nn.Linear(3 * self.d_l, 3 * self.d_l)
        
        # 8. final project
        self.proj1 = nn.Linear(combined_dim, combined_dim)
        self.proj2 = nn.Linear(combined_dim, combined_dim)
        self.out_layer = nn.Linear(combined_dim, output_dim)
        
    def get_network(self, self_type='l', layers=-1):
        if self_type in ['l', 'al', 'vl']:
            embed_dim, attn_dropout = self.d_l, self.attn_dropout
        elif self_type in ['a', 'la', 'va']:
            embed_dim, attn_dropout = self.d_a, self.attn_dropout_a
        elif self_type in ['v', 'lv', 'av']:
            embed_dim, attn_dropout = self.d_v, self.attn_dropout_v        
        elif self_type == 'l_mem':
            embed_dim, attn_dropout = self.d_l, self.attn_dropout
        elif self_type == 'a_mem':
            embed_dim, attn_dropout = self.d_a, self.attn_dropout
        elif self_type == 'v_mem':
            embed_dim, attn_dropout = self.d_v, self.attn_dropout
        else:
            raise ValueError("Unknown network type")

        return TransformerEncoder(embed_dim=embed_dim,
                                  num_heads=self.num_heads,
                                  layers=max(self.layers, layers),
                                  attn_dropout=attn_dropout,
                                  relu_dropout=self.relu_dropout,
                                  res_dropout=self.res_dropout,
                                  embed_dropout=self.embed_dropout,
                                  attn_mask=self.attn_mask)


    def forward(
        self,
        text,
        audio,
        video,
        fusion_residual=None,
        availability_mask=None,
    ):
        if availability_mask is not None:
            if availability_mask.ndim != 2 or availability_mask.size(1) != 3:
                raise ValueError("availability_mask must have shape [batch, 3].")
            if availability_mask.size(0) != audio.size(0):
                raise ValueError("availability_mask batch dimension mismatch.")
            availability_mask = availability_mask.to(device=audio.device, dtype=audio.dtype)
            if not torch.all(availability_mask[:, 0] == 1):
                raise ValueError("Language must remain available in SAFE-DLF.")
            availability_l = availability_mask[:, 0]
            availability_a = availability_mask[:, 1]
            availability_v = availability_mask[:, 2]
        else:
            availability_l = availability_a = availability_v = None

        def mask_sequence(value, availability):
            if availability is None:
                return value
            return value * availability.to(value).view(1, -1, 1)

        def mask_batch(value, availability):
            if availability is None:
                return value
            return value * availability.to(value).view(
                value.size(0), *([1] * (value.ndim - 1))
            )

        def encode(network, query, availability, key=None, value=None):
            if key is None:
                if availability is None:
                    return network(query)
                return network(query, availability_mask=availability)
            if availability is None or bool(torch.all(availability > 0.5)):
                return network(query, key, value)
            present = torch.nonzero(availability > 0.5, as_tuple=False).view(-1)
            result = query.new_zeros(query.shape)
            if present.numel() == 0:
                return result
            encoded = network(
                query.index_select(1, present),
                key.index_select(1, present),
                value.index_select(1, present),
            )
            return result.index_copy(1, present, encoded)

        #extraction
        if self.use_bert:
            text = self.text_model(text)
        x_l = F.dropout(text.transpose(1, 2), p=self.text_dropout, training=self.training) 
        x_a = audio.transpose(1, 2)
        x_v = video.transpose(1, 2)
        

        proj_x_l = x_l if self.orig_d_l == self.d_l else self.proj_l(x_l) 
        proj_x_a = x_a if self.orig_d_a == self.d_a else self.proj_a(x_a) 
        proj_x_v = x_v if self.orig_d_v == self.d_v else self.proj_v(x_v)
        
        proj_x_l = proj_x_l.permute(2, 0, 1)
        proj_x_v = proj_x_v .permute(2, 0, 1)
        proj_x_a = proj_x_a.permute(2, 0, 1)
        proj_x_l = mask_sequence(proj_x_l, availability_l)
        proj_x_v = mask_sequence(proj_x_v, availability_v)
        proj_x_a = mask_sequence(proj_x_a, availability_a)

        #disentanglement
        s_l = encode(self.encoder_s_l, proj_x_l, availability_l)
        s_v = encode(self.encoder_s_v, proj_x_v, availability_v)
        s_a = encode(self.encoder_s_a, proj_x_a, availability_a)

        c_l = encode(self.encoder_c, proj_x_l, availability_l)
        c_v = encode(self.encoder_c, proj_x_v, availability_v)
        c_a = encode(self.encoder_c, proj_x_a, availability_a)


        s_l = s_l.permute(1, 2, 0)   
        s_v = s_v.permute(1, 2, 0)
        s_a = s_a.permute(1, 2, 0)

        c_l = c_l.permute(1, 2, 0)
        c_v = c_v.permute(1, 2, 0)
        c_a = c_a.permute(1, 2, 0)
        c_list = [c_l, c_v, c_a]


        c_l_sim = mask_batch(
            self.align_c_l(c_l.contiguous().view(x_l.size(0), -1)),
            availability_l,
        )
        c_v_sim = mask_batch(
            self.align_c_v(c_v.contiguous().view(x_l.size(0), -1)),
            availability_v,
        )
        c_a_sim = mask_batch(
            self.align_c_a(c_a.contiguous().view(x_l.size(0), -1)),
            availability_a,
        )
        
        recon_l = mask_batch(
            self.decoder_l(torch.cat([s_l, c_list[0]], dim=1)), availability_l
        )
        recon_v = mask_batch(
            self.decoder_v(torch.cat([s_v, c_list[1]], dim=1)), availability_v
        )
        recon_a = mask_batch(
            self.decoder_a(torch.cat([s_a, c_list[2]], dim=1)), availability_a
        )

        recon_l = recon_l.permute(2, 0, 1)  
        recon_v = recon_v.permute(2, 0, 1)   
        recon_a = recon_a.permute(2, 0, 1)

        s_l_r = encode(self.encoder_s_l, recon_l, availability_l).permute(1, 2, 0)
        s_v_r = encode(self.encoder_s_v, recon_v, availability_v).permute(1, 2, 0)
        s_a_r = encode(self.encoder_s_a, recon_a, availability_a).permute(1, 2, 0)
        
        s_l = s_l.permute(2, 0, 1)  
        s_v = s_v.permute(2, 0, 1)   
        s_a = s_a.permute(2, 0, 1)

        c_l = c_l.permute(2, 0, 1)
        c_v = c_v.permute(2, 0, 1)
        c_a = c_a.permute(2, 0, 1)
       
       #enhancement
        hs_l_low = c_l.transpose(0, 1).contiguous().view(x_l.size(0), -1)  
        repr_l_low = mask_batch(self.proj1_l_low(hs_l_low), availability_l)
        hs_proj_l_low = mask_batch(
            self.proj2_l_low(
                F.dropout(
                    F.relu(repr_l_low, inplace=True),
                    p=self.output_dropout,
                    training=self.training,
                )
            ),
            availability_l,
        )
        hs_proj_l_low = mask_batch(hs_proj_l_low + hs_l_low, availability_l)
        logits_l_low = mask_batch(
            self.out_layer_l_low(hs_proj_l_low), availability_l
        )

        hs_v_low = c_v.transpose(0, 1).contiguous().view(x_v.size(0), -1)
        repr_v_low = mask_batch(self.proj1_v_low(hs_v_low), availability_v)
        hs_proj_v_low = mask_batch(
            self.proj2_v_low(
                F.dropout(
                    F.relu(repr_v_low, inplace=True),
                    p=self.output_dropout,
                    training=self.training,
                )
            ),
            availability_v,
        )
        hs_proj_v_low = mask_batch(hs_proj_v_low + hs_v_low, availability_v)
        logits_v_low = mask_batch(
            self.out_layer_v_low(hs_proj_v_low), availability_v
        )

        hs_a_low = c_a.transpose(0, 1).contiguous().view(x_a.size(0), -1)
        repr_a_low = mask_batch(self.proj1_a_low(hs_a_low), availability_a)
        hs_proj_a_low = mask_batch(
            self.proj2_a_low(
                F.dropout(
                    F.relu(repr_a_low, inplace=True),
                    p=self.output_dropout,
                    training=self.training,
                )
            ),
            availability_a,
        )
        hs_proj_a_low = mask_batch(hs_proj_a_low + hs_a_low, availability_a)
        logits_a_low = mask_batch(
            self.out_layer_a_low(hs_proj_a_low), availability_a
        )

        
        c_l_att = encode(self.self_attentions_c_l, c_l, availability_l)
        if type(c_l_att) == tuple:
            c_l_att = c_l_att[0]
        c_l_att = c_l_att[-1]

        c_v_att = encode(self.self_attentions_c_v, c_v, availability_v)
        if type(c_v_att) == tuple:
            c_v_att = c_v_att[0]
        c_v_att = c_v_att[-1]

        c_a_att = encode(self.self_attentions_c_a, c_a, availability_a)
        if type(c_a_att) == tuple:
            c_a_att = c_a_att[0]
        c_a_att = c_a_att[-1]

        c_fusion = torch.cat([c_l_att, c_v_att, c_a_att], dim=1)   

        c_proj = self.proj2_c(
            F.dropout(F.relu(self.proj1_c(c_fusion), inplace=True), p=self.output_dropout,
                      training=self.training))
        c_proj += c_fusion                       
        logits_c = self.out_layer_c(c_proj)     
        
        # LFA
        # L --> L                
        h_ls = s_l                     
        h_ls = encode(self.trans_l_mem, h_ls, availability_l)
        if type(h_ls) == tuple:
            h_ls = h_ls[0]
        last_h_l = last_hs = h_ls[-1]  

        # A --> L
        h_l_with_as = encode(
            self.trans_l_with_a, s_l, availability_a, key=s_a, value=s_a
        )
        h_as = h_l_with_as
        h_as = encode(self.trans_a_mem, h_as, availability_a)
        if type(h_as) == tuple:
            h_as = h_as[0]
        last_h_a = last_hs = h_as[-1]  

        # V --> L
        h_l_with_vs = encode(
            self.trans_l_with_v, s_l, availability_v, key=s_v, value=s_v
        )
        h_vs = h_l_with_vs
        h_vs = encode(self.trans_v_mem, h_vs, availability_v)
        if type(h_vs) == tuple:
            h_vs = h_vs[0]
        last_h_v = last_hs = h_vs[-1]  


        hs_proj_l_high = mask_batch(
            self.proj2_l_high(
                F.dropout(
                    F.relu(self.proj1_l_high(last_h_l), inplace=True),
                    p=self.output_dropout,
                    training=self.training,
                )
            ),
            availability_l,
        )
        hs_proj_l_high = mask_batch(hs_proj_l_high + last_h_l, availability_l)
        logits_l_high = mask_batch(
            self.out_layer_l_high(hs_proj_l_high), availability_l
        )

        hs_proj_v_high = mask_batch(
            self.proj2_v_high(
                F.dropout(
                    F.relu(self.proj1_v_high(last_h_v), inplace=True),
                    p=self.output_dropout,
                    training=self.training,
                )
            ),
            availability_v,
        )
        hs_proj_v_high = mask_batch(hs_proj_v_high + last_h_v, availability_v)
        logits_v_high = mask_batch(
            self.out_layer_v_high(hs_proj_v_high), availability_v
        )

        hs_proj_a_high = mask_batch(
            self.proj2_a_high(
                F.dropout(
                    F.relu(self.proj1_a_high(last_h_a), inplace=True),
                    p=self.output_dropout,
                    training=self.training,
                )
            ),
            availability_a,
        )
        hs_proj_a_high = mask_batch(hs_proj_a_high + last_h_a, availability_a)
        logits_a_high = mask_batch(
            self.out_layer_a_high(hs_proj_a_high), availability_a
        )
        
        #fusion
        last_h_l = mask_batch(
            torch.sigmoid(self.projector_l(hs_proj_l_high)), availability_l
        )
        last_h_v = mask_batch(
            torch.sigmoid(self.projector_v(hs_proj_v_high)), availability_v
        )
        last_h_a = mask_batch(
            torch.sigmoid(self.projector_a(hs_proj_a_high)), availability_a
        )
        c_fusion = torch.sigmoid(self.projector_c(c_fusion))
        
        last_hs = torch.cat([last_h_l, last_h_v, last_h_a, c_fusion], dim=1)   
        fusion_input = last_hs


        # This optional residual leaves the original DLF path unchanged by default.
        if fusion_residual is not None:
            last_hs = last_hs + fusion_residual
        #prediction
        last_hs_proj = self.proj2(
            F.dropout(F.relu(self.proj1(last_hs), inplace=True), p=self.output_dropout, training=self.training))
        last_hs_proj += last_hs                          

        output = self.out_layer(last_hs_proj)

        res = {
            'origin_l': proj_x_l,
            'origin_v': proj_x_v,
            'origin_a': proj_x_a,
            's_l': s_l,        
            's_v': s_v,
            's_a': s_a,
            'c_l': c_l,
            'c_v': c_v,
            'c_a': c_a,
            's_l_r': s_l_r,
            's_v_r': s_v_r,
            's_a_r': s_a_r,
            'recon_l': recon_l,
            'recon_v': recon_v,
            'recon_a': recon_a,
            'c_l_sim': c_l_sim,
            'c_v_sim': c_v_sim,
            'c_a_sim': c_a_sim,
            'logits_l_hetero': logits_l_high,           
            'logits_v_hetero': logits_v_high, 
            'logits_a_hetero': logits_a_high,
            'logits_c': logits_c,
            'output_logit': output,
            'lfa_l': last_h_l,
            'lfa_v': last_h_v,
            'lfa_a': last_h_a,
            'lfa_cross_v': h_l_with_vs,
            'lfa_cross_a': h_l_with_as,
            'lfa_ffn_v': h_vs,
            'lfa_ffn_a': h_as,
            'specific_hidden_l': hs_proj_l_high,
            'specific_hidden_v': hs_proj_v_high,
            'specific_hidden_a': hs_proj_a_high,
            'fusion_input': fusion_input,
        }
        return res
