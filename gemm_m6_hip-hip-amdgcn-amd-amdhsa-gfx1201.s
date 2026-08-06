	.amdgcn_target "amdgcn-amd-amdhsa--gfx1201"
	.amdhsa_code_object_version 6
	.text
	.protected	_Z14plain_direct16PK14__hip_fp8_e4m3S1_Pfiii ; -- Begin function _Z14plain_direct16PK14__hip_fp8_e4m3S1_Pfiii
	.globl	_Z14plain_direct16PK14__hip_fp8_e4m3S1_Pfiii
	.p2align	8
	.type	_Z14plain_direct16PK14__hip_fp8_e4m3S1_Pfiii,@function
_Z14plain_direct16PK14__hip_fp8_e4m3S1_Pfiii: ; @_Z14plain_direct16PK14__hip_fp8_e4m3S1_Pfiii
; %bb.0:
	s_clause 0x2
	s_load_b64 s[2:3], s[0:1], 0x1c
	s_load_b128 s[4:7], s[0:1], 0x0
	s_load_b64 s[0:1], s[0:1], 0x10
	v_and_b32_e32 v10, 15, v0
	v_lshrrev_b32_e32 v11, 1, v0
	s_mov_b32 s8, 0
	s_wait_kmcnt 0x0
	s_cmp_gt_i32 s3, 0
	s_cbranch_scc1 .LBB0_2
; %bb.1:                                ; %.._crit_edge_crit_edge
	v_and_b32_e32 v9, 15, v0
	v_and_b32_e32 v8, 8, v11
	s_branch .LBB0_3
.LBB0_2:
	s_mov_b32 s8, -1
                                        ; implicit-def: $vgpr9
                                        ; implicit-def: $vgpr8
.LBB0_3:                                ; %Flow
	v_dual_mov_b32 v7, 0 :: v_dual_mov_b32 v6, 0
	v_dual_mov_b32 v5, 0 :: v_dual_mov_b32 v4, 0
	v_dual_mov_b32 v3, 0 :: v_dual_mov_b32 v2, 0
	v_dual_mov_b32 v1, 0 :: v_dual_mov_b32 v0, 0
	s_and_not1_b32 vcc_lo, exec_lo, s8
	s_cbranch_vccnz .LBB0_7
; %bb.4:                                ; %.lr.ph
	v_mov_b32_e32 v0, 0
	v_and_b32_e32 v8, 8, v11
	s_mul_i32 s8, ttmp7, s3
	s_mul_i32 s9, ttmp9, s3
	s_lshl_b32 s8, s8, 4
	s_lshl_b32 s10, s9, 4
	v_mad_co_u64_u32 v[1:2], null, s3, v10, v[8:9]
	s_ashr_i32 s9, s8, 31
	s_ashr_i32 s11, s10, 31
	s_add_nc_u64 s[4:5], s[4:5], s[8:9]
	s_add_nc_u64 s[6:7], s[6:7], s[10:11]
	v_dual_mov_b32 v2, v0 :: v_dual_mov_b32 v3, v0
	v_dual_mov_b32 v4, v0 :: v_dual_mov_b32 v5, v0
	v_add_co_u32 v9, s4, s4, v1
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v11, null, s5, 0, s4
	v_add_co_u32 v12, s4, s6, v1
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v13, null, s7, 0, s4
	v_dual_mov_b32 v1, v0 :: v_dual_mov_b32 v6, v0
	v_mov_b32_e32 v7, v0
	s_mov_b64 s[4:5], 0
.LBB0_5:                                ; =>This Inner Loop Header: Depth=1
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v14, vcc_lo, v9, s4
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v15, null, s5, v11, vcc_lo
	v_add_co_u32 v16, vcc_lo, v12, s4
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v17, null, s5, v13, vcc_lo
	global_load_b64 v[14:15], v[14:15], off
	global_load_b64 v[16:17], v[16:17], off
	s_add_nc_u64 s[4:5], s[4:5], 16
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_ge_i32 s4, s3
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x16_fp8_fp8 v[0:7], v[14:15], v[16:17], v[0:7]
	s_cbranch_scc0 .LBB0_5
; %bb.6:                                ; %._crit_edge.loopexit
	v_mov_b32_e32 v9, v10
.LBB0_7:                                ; %Flow79
	s_delay_alu instid0(VALU_DEP_1)
	v_mad_co_u64_u32 v[8:9], null, s2, v8, v[9:10]
	v_mov_b32_e32 v9, 0
	s_mul_i32 s3, ttmp7, s2
	s_lshl_b32 s6, ttmp9, 4
	s_wait_alu depctr_sa_sdst(0)
	s_lshl_b32 s4, s3, 4
	s_ashr_i32 s7, s6, 31
	s_wait_alu depctr_sa_sdst(0)
	s_ashr_i32 s5, s4, 31
	s_lshl_b64 s[6:7], s[6:7], 2
	v_lshlrev_b64_e32 v[10:11], 2, v[8:9]
	v_add_nc_u32_e32 v8, s2, v8
	s_wait_alu depctr_sa_sdst(0)
	s_lshl_b64 s[4:5], s[4:5], 2
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[0:1], s[0:1], s[4:5]
	v_lshlrev_b64_e32 v[12:13], 2, v[8:9]
	v_add_nc_u32_e32 v8, s2, v8
	s_add_nc_u64 s[0:1], s[0:1], s[6:7]
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_u32 v10, vcc_lo, s0, v10
	v_lshlrev_b64_e32 v[14:15], 2, v[8:9]
	v_add_nc_u32_e32 v8, s2, v8
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v11, null, s1, v11, vcc_lo
	v_add_co_u32 v12, vcc_lo, s0, v12
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v13, null, s1, v13, vcc_lo
	v_add_co_u32 v14, vcc_lo, s0, v14
	v_lshlrev_b64_e32 v[16:17], 2, v[8:9]
	v_add_nc_u32_e32 v8, s2, v8
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v15, null, s1, v15, vcc_lo
	s_clause 0x2
	global_store_b32 v[10:11], v0, off
	global_store_b32 v[12:13], v1, off
	global_store_b32 v[14:15], v2, off
	v_lshlrev_b64_e32 v[0:1], 2, v[8:9]
	v_add_nc_u32_e32 v8, s2, v8
	v_add_co_u32 v10, vcc_lo, s0, v16
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v11, null, s1, v17, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_4) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[12:13], 2, v[8:9]
	v_add_nc_u32_e32 v8, s2, v8
	v_add_co_u32 v0, vcc_lo, s0, v0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v1, null, s1, v1, vcc_lo
	v_lshlrev_b64_e32 v[14:15], 2, v[8:9]
	v_add_nc_u32_e32 v8, s2, v8
	v_add_co_u32 v12, vcc_lo, s0, v12
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v13, null, s1, v13, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_3) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[8:9], 2, v[8:9]
	v_add_co_u32 v14, vcc_lo, s0, v14
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v15, null, s1, v15, vcc_lo
	v_add_co_u32 v8, vcc_lo, s0, v8
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, s1, v9, vcc_lo
	s_clause 0x4
	global_store_b32 v[10:11], v3, off
	global_store_b32 v[0:1], v4, off
	global_store_b32 v[12:13], v5, off
	global_store_b32 v[14:15], v6, off
	global_store_b32 v[8:9], v7, off
	s_endpgm
.Lfunc_end0:
	.size	_Z14plain_direct16PK14__hip_fp8_e4m3S1_Pfiii, .Lfunc_end0-_Z14plain_direct16PK14__hip_fp8_e4m3S1_Pfiii
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel _Z14plain_direct16PK14__hip_fp8_e4m3S1_Pfiii
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 36
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_dispatch_ptr 0
		.amdhsa_user_sgpr_queue_ptr 0
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 0
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_wavefront_size32 1
		.amdhsa_uses_dynamic_stack 0
		.amdhsa_enable_private_segment 0
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 1
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 18
		.amdhsa_next_free_sgpr 12
		.amdhsa_reserve_vcc 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_fp16_overflow 0
		.amdhsa_workgroup_processor_mode 1
		.amdhsa_memory_ordered 1
		.amdhsa_forward_progress 1
		.amdhsa_inst_pref_size 6
		.amdhsa_round_robin_scheduling 0
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.text
                                        ; -- End function
	.set .L_Z14plain_direct16PK14__hip_fp8_e4m3S1_Pfiii.num_vgpr, 18
	.set .L_Z14plain_direct16PK14__hip_fp8_e4m3S1_Pfiii.num_agpr, 0
	.set .L_Z14plain_direct16PK14__hip_fp8_e4m3S1_Pfiii.numbered_sgpr, 12
	.set .L_Z14plain_direct16PK14__hip_fp8_e4m3S1_Pfiii.num_named_barrier, 0
	.set .L_Z14plain_direct16PK14__hip_fp8_e4m3S1_Pfiii.private_seg_size, 0
	.set .L_Z14plain_direct16PK14__hip_fp8_e4m3S1_Pfiii.uses_vcc, 1
	.set .L_Z14plain_direct16PK14__hip_fp8_e4m3S1_Pfiii.uses_flat_scratch, 0
	.set .L_Z14plain_direct16PK14__hip_fp8_e4m3S1_Pfiii.has_dyn_sized_stack, 0
	.set .L_Z14plain_direct16PK14__hip_fp8_e4m3S1_Pfiii.has_recursion, 0
	.set .L_Z14plain_direct16PK14__hip_fp8_e4m3S1_Pfiii.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 736
; TotalNumSgprs: 14
; NumVgprs: 18
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 0
; VGPRBlocks: 2
; NumSGPRsForWavesPerEU: 14
; NumVGPRsForWavesPerEU: 18
; Occupancy: 16
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 2
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
	.text
	.protected	_Z14plain_direct32PK14__hip_fp8_e4m3S1_Pfiii ; -- Begin function _Z14plain_direct32PK14__hip_fp8_e4m3S1_Pfiii
	.globl	_Z14plain_direct32PK14__hip_fp8_e4m3S1_Pfiii
	.p2align	8
	.type	_Z14plain_direct32PK14__hip_fp8_e4m3S1_Pfiii,@function
_Z14plain_direct32PK14__hip_fp8_e4m3S1_Pfiii: ; @_Z14plain_direct32PK14__hip_fp8_e4m3S1_Pfiii
; %bb.0:
	s_clause 0x2
	s_load_b64 s[2:3], s[0:1], 0x1c
	s_load_b128 s[4:7], s[0:1], 0x0
	s_load_b64 s[0:1], s[0:1], 0x10
	v_and_b32_e32 v13, 15, v0
	s_mov_b32 s8, 0
	s_wait_kmcnt 0x0
	s_cmp_gt_i32 s3, 0
	s_cbranch_scc1 .LBB1_2
; %bb.1:                                ; %.._crit_edge_crit_edge
	v_and_b32_e32 v9, 15, v0
	s_branch .LBB1_3
.LBB1_2:
	s_mov_b32 s8, -1
                                        ; implicit-def: $vgpr9
.LBB1_3:                                ; %Flow
	v_dual_mov_b32 v8, 0 :: v_dual_mov_b32 v7, 0
	v_dual_mov_b32 v6, 0 :: v_dual_mov_b32 v5, 0
	v_dual_mov_b32 v4, 0 :: v_dual_mov_b32 v3, 0
	v_dual_mov_b32 v2, 0 :: v_dual_mov_b32 v1, 0
	s_and_not1_b32 vcc_lo, exec_lo, s8
	s_cbranch_vccnz .LBB1_7
; %bb.4:                                ; %.lr.ph
	v_and_b32_e32 v1, 16, v0
	s_mul_i32 s8, ttmp9, s3
	s_mul_i32 s9, ttmp7, s3
	s_lshl_b32 s8, s8, 4
	s_lshl_b32 s10, s9, 4
	v_mad_co_u64_u32 v[2:3], null, s3, v13, v[1:2]
	s_ashr_i32 s9, s8, 31
	s_ashr_i32 s11, s10, 31
	s_add_nc_u64 s[6:7], s[6:7], s[8:9]
	s_add_nc_u64 s[4:5], s[4:5], s[10:11]
	v_mov_b32_e32 v1, 0
	v_add_co_u32 v3, s6, s6, v2
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v4, null, s7, 0, s6
	v_add_co_u32 v2, s4, s4, v2
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v5, null, s5, 0, s4
	v_add_co_u32 v9, vcc_lo, v3, 8
	s_delay_alu instid0(VALU_DEP_1)
	v_add_co_ci_u32_e64 v10, null, 0, v4, vcc_lo
	v_add_co_u32 v11, vcc_lo, v2, 8
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v12, null, 0, v5, vcc_lo
	v_dual_mov_b32 v2, v1 :: v_dual_mov_b32 v3, v1
	v_dual_mov_b32 v4, v1 :: v_dual_mov_b32 v5, v1
	v_dual_mov_b32 v6, v1 :: v_dual_mov_b32 v7, v1
	v_mov_b32_e32 v8, v1
	s_mov_b32 s4, 0
.LBB1_5:                                ; =>This Inner Loop Header: Depth=1
	global_load_b128 v[14:17], v[11:12], off offset:-8
	global_load_b128 v[18:21], v[9:10], off offset:-8
	v_add_co_u32 v9, vcc_lo, v9, 32
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v10, null, 0, v10, vcc_lo
	v_add_co_u32 v11, vcc_lo, v11, 32
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v12, null, 0, v12, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s4, s4, 32
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_ge_i32 s4, s3
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x16_fp8_fp8 v[1:8], v[14:15], v[18:19], v[1:8]
	s_delay_alu instid0(VALU_DEP_1)
	v_wmma_f32_16x16x16_fp8_fp8 v[1:8], v[16:17], v[20:21], v[1:8]
	s_cbranch_scc0 .LBB1_5
; %bb.6:                                ; %._crit_edge.loopexit
	v_mov_b32_e32 v9, v13
.LBB1_7:                                ; %Flow83
	v_lshrrev_b32_e32 v0, 1, v0
	s_mul_i32 s3, ttmp7, s2
	s_lshl_b32 s6, ttmp9, 4
	s_wait_alu depctr_sa_sdst(0)
	s_lshl_b32 s4, s3, 4
	s_ashr_i32 s7, s6, 31
	v_and_b32_e32 v0, 8, v0
	s_wait_alu depctr_sa_sdst(0)
	s_ashr_i32 s5, s4, 31
	s_lshl_b64 s[6:7], s[6:7], 2
	s_wait_alu depctr_sa_sdst(0)
	s_lshl_b64 s[4:5], s[4:5], 2
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[0:1], s[0:1], s[4:5]
	v_mad_co_u64_u32 v[9:10], null, s2, v0, v[9:10]
	v_mov_b32_e32 v10, 0
	s_add_nc_u64 s[0:1], s[0:1], s[6:7]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[11:12], 2, v[9:10]
	v_add_nc_u32_e32 v9, s2, v9
	v_lshlrev_b64_e32 v[13:14], 2, v[9:10]
	v_add_nc_u32_e32 v9, s2, v9
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_add_co_u32 v11, vcc_lo, s0, v11
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v12, null, s1, v12, vcc_lo
	v_lshlrev_b64_e32 v[15:16], 2, v[9:10]
	v_add_nc_u32_e32 v9, s2, v9
	v_add_co_u32 v13, vcc_lo, s0, v13
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v14, null, s1, v14, vcc_lo
	s_delay_alu instid0(VALU_DEP_3)
	v_lshlrev_b64_e32 v[17:18], 2, v[9:10]
	v_add_nc_u32_e32 v9, s2, v9
	s_clause 0x1
	global_store_b32 v[11:12], v1, off
	global_store_b32 v[13:14], v2, off
	v_add_co_u32 v15, vcc_lo, s0, v15
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v16, null, s1, v16, vcc_lo
	v_lshlrev_b64_e32 v[0:1], 2, v[9:10]
	v_add_nc_u32_e32 v9, s2, v9
	v_add_co_u32 v2, vcc_lo, s0, v17
	global_store_b32 v[15:16], v3, off
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v3, null, s1, v18, vcc_lo
	v_lshlrev_b64_e32 v[11:12], 2, v[9:10]
	v_add_nc_u32_e32 v9, s2, v9
	v_add_co_u32 v0, vcc_lo, s0, v0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v1, null, s1, v1, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_4) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[13:14], 2, v[9:10]
	v_add_nc_u32_e32 v9, s2, v9
	v_add_co_u32 v11, vcc_lo, s0, v11
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v12, null, s1, v12, vcc_lo
	v_lshlrev_b64_e32 v[9:10], 2, v[9:10]
	v_add_co_u32 v13, vcc_lo, s0, v13
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v14, null, s1, v14, vcc_lo
	s_delay_alu instid0(VALU_DEP_3)
	v_add_co_u32 v9, vcc_lo, s0, v9
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v10, null, s1, v10, vcc_lo
	s_clause 0x4
	global_store_b32 v[2:3], v4, off
	global_store_b32 v[0:1], v5, off
	global_store_b32 v[11:12], v6, off
	global_store_b32 v[13:14], v7, off
	global_store_b32 v[9:10], v8, off
	s_endpgm
.Lfunc_end1:
	.size	_Z14plain_direct32PK14__hip_fp8_e4m3S1_Pfiii, .Lfunc_end1-_Z14plain_direct32PK14__hip_fp8_e4m3S1_Pfiii
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel _Z14plain_direct32PK14__hip_fp8_e4m3S1_Pfiii
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 36
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_dispatch_ptr 0
		.amdhsa_user_sgpr_queue_ptr 0
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 0
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_wavefront_size32 1
		.amdhsa_uses_dynamic_stack 0
		.amdhsa_enable_private_segment 0
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 1
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 22
		.amdhsa_next_free_sgpr 12
		.amdhsa_reserve_vcc 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_fp16_overflow 0
		.amdhsa_workgroup_processor_mode 1
		.amdhsa_memory_ordered 1
		.amdhsa_forward_progress 1
		.amdhsa_inst_pref_size 7
		.amdhsa_round_robin_scheduling 0
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.text
                                        ; -- End function
	.set .L_Z14plain_direct32PK14__hip_fp8_e4m3S1_Pfiii.num_vgpr, 22
	.set .L_Z14plain_direct32PK14__hip_fp8_e4m3S1_Pfiii.num_agpr, 0
	.set .L_Z14plain_direct32PK14__hip_fp8_e4m3S1_Pfiii.numbered_sgpr, 12
	.set .L_Z14plain_direct32PK14__hip_fp8_e4m3S1_Pfiii.num_named_barrier, 0
	.set .L_Z14plain_direct32PK14__hip_fp8_e4m3S1_Pfiii.private_seg_size, 0
	.set .L_Z14plain_direct32PK14__hip_fp8_e4m3S1_Pfiii.uses_vcc, 1
	.set .L_Z14plain_direct32PK14__hip_fp8_e4m3S1_Pfiii.uses_flat_scratch, 0
	.set .L_Z14plain_direct32PK14__hip_fp8_e4m3S1_Pfiii.has_dyn_sized_stack, 0
	.set .L_Z14plain_direct32PK14__hip_fp8_e4m3S1_Pfiii.has_recursion, 0
	.set .L_Z14plain_direct32PK14__hip_fp8_e4m3S1_Pfiii.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 792
; TotalNumSgprs: 14
; NumVgprs: 22
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 0
; VGPRBlocks: 2
; NumSGPRsForWavesPerEU: 14
; NumVGPRsForWavesPerEU: 22
; Occupancy: 16
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 2
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
	.text
	.protected	_Z12fused_m5_k32PK14__hip_fp8_e4m3PKhS3_Pfiiii ; -- Begin function _Z12fused_m5_k32PK14__hip_fp8_e4m3PKhS3_Pfiiii
	.globl	_Z12fused_m5_k32PK14__hip_fp8_e4m3PKhS3_Pfiiii
	.p2align	8
	.type	_Z12fused_m5_k32PK14__hip_fp8_e4m3PKhS3_Pfiiii,@function
_Z12fused_m5_k32PK14__hip_fp8_e4m3PKhS3_Pfiiii: ; @_Z12fused_m5_k32PK14__hip_fp8_e4m3PKhS3_Pfiiii
; %bb.0:
	s_load_b256 s[4:11], s[0:1], 0x0
	s_mov_b32 s3, exec_lo
	v_cmpx_gt_u32_e32 0x100, v0
	s_cbranch_execz .LBB2_5
; %bb.1:                                ; %.lr.ph.preheader
	v_lshlrev_b32_e32 v3, 3, v0
	s_wait_kmcnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v1, s2, s8, v3
	v_add_co_ci_u32_e64 v2, null, s9, 0, s2
	s_mov_b32 s8, 0
	s_branch .LBB2_3
.LBB2_2:                                ;   in Loop: Header=BB2_3 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_add_nc_u32_e32 v4, 0x100, v3
	v_cmp_lt_u32_e32 vcc_lo, 0x6ff, v3
	v_add_co_u32 v1, s2, 0x100, v1
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v2, null, 0, v2, s2
	v_mov_b32_e32 v3, v4
	s_or_b32 s8, vcc_lo, s8
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 exec_lo, exec_lo, s8
	s_cbranch_execz .LBB2_5
.LBB2_3:                                ; %.lr.ph
                                        ; =>This Inner Loop Header: Depth=1
	s_mov_b32 s2, exec_lo
	v_cmpx_gt_u32_e32 0x7f9, v3
	s_cbranch_execz .LBB2_2
; %bb.4:                                ;   in Loop: Header=BB2_3 Depth=1
	global_load_b64 v[4:5], v[1:2], off
	s_wait_loadcnt 0x0
	ds_store_b64 v3, v[4:5]
	s_branch .LBB2_2
.LBB2_5:                                ; %Flow235
	s_or_b32 exec_lo, exec_lo, s3
	s_load_b96 s[12:14], s[0:1], 0x24
	s_wait_dscnt 0x0
	s_barrier_signal -1
	v_and_b32_e32 v13, 15, v0
	v_lshrrev_b32_e32 v14, 1, v0
	s_mov_b32 s0, 0
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_kmcnt 0x0
	s_cmp_gt_i32 s13, 0
	s_cbranch_scc1 .LBB2_7
; %bb.6:                                ; %._crit_edge.._crit_edge102_crit_edge
	v_and_b32_e32 v9, 15, v0
	v_lshrrev_b32_e32 v10, 1, v0
	s_branch .LBB2_8
.LBB2_7:
	s_mov_b32 s0, -1
                                        ; implicit-def: $vgpr9
                                        ; implicit-def: $vgpr10
.LBB2_8:                                ; %Flow232
	v_dual_mov_b32 v8, 0 :: v_dual_mov_b32 v7, 0
	v_dual_mov_b32 v6, 0 :: v_dual_mov_b32 v5, 0
	v_dual_mov_b32 v4, 0 :: v_dual_mov_b32 v3, 0
	v_dual_mov_b32 v2, 0 :: v_dual_mov_b32 v1, 0
	s_and_not1_b32 vcc_lo, exec_lo, s0
	s_lshl_b32 s2, ttmp9, 4
	s_cbranch_vccnz .LBB2_20
; %bb.9:                                ; %.lr.ph101
	s_wait_alu depctr_sa_sdst(0)
	v_add_nc_u32_e32 v1, s2, v14
	v_and_b32_e32 v2, 16, v0
	s_mul_i32 s0, ttmp7, s13
	v_lshlrev_b32_e32 v6, 5, v13
	s_lshl_b32 s8, s0, 4
	v_cmp_gt_i32_e32 vcc_lo, s12, v1
	v_mul_lo_u32 v15, s14, v1
	v_mov_b32_e32 v1, 0
	v_mad_co_u64_u32 v[3:4], null, s13, v13, v[2:3]
	s_wait_alu depctr_sa_sdst(0)
	s_ashr_i32 s9, s8, 31
	v_lshlrev_b32_e32 v5, 4, v0
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[4:5], s[4:5], s[8:9]
	v_or3_b32 v17, v6, v2, 0x800
	v_cmp_gt_u32_e64 s0, 32, v0
	v_and_b32_e32 v0, 1, v0
	v_lshl_add_u32 v16, v14, 5, 0x800
	v_add_co_u32 v2, s1, s4, v3
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_add_co_ci_u32_e64 v3, null, s5, 0, s1
	v_mov_b32_e32 v4, v1
	v_add_co_u32 v9, s1, v2, 8
	v_mov_b32_e32 v2, v1
	v_and_b32_e32 v18, 16, v5
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v10, null, 0, v3, s1
	v_mov_b32_e32 v3, v1
	v_dual_mov_b32 v5, v1 :: v_dual_lshlrev_b32 v0, 3, v0
	v_mov_b32_e32 v7, v1
	v_dual_mov_b32 v19, v18 :: v_dual_mov_b32 v6, v1
	v_mov_b32_e32 v8, v1
	s_mov_b32 s3, 0
	s_mov_b32 s4, 0
	s_mov_b32 s5, 0
	s_branch .LBB2_12
.LBB2_10:                               ; %.loopexit.1
                                        ;   in Loop: Header=BB2_12 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s14
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_2)
	v_perm_b32 v21, v23, v24, 0xc0c0004
	s_wait_dscnt 0x0
	v_perm_b32 v12, v22, v12, 0xc0c0004
	v_add_nc_u32_e32 v20, v20, v18
	v_lshl_or_b32 v12, v12, 16, v21
	ds_store_b64 v20, v[11:12] offset:8
.LBB2_11:                               ; %Flow
                                        ;   in Loop: Header=BB2_12 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s9
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_lshl_add_u32 v11, s8, 9, v17
	v_add_nc_u32_e32 v0, 16, v0
	v_add_nc_u32_e32 v19, 32, v19
	s_add_co_i32 s5, s5, 32
	s_add_co_i32 s4, s4, 1
	s_add_co_i32 s3, s3, 16
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_ge_i32 s5, s13
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	global_load_b128 v[20:23], v[9:10], off offset:-8
	ds_load_b128 v[24:27], v11
	v_add_co_u32 v9, s1, v9, 32
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v10, null, 0, v10, s1
	s_wait_loadcnt_dscnt 0x0
	v_wmma_f32_16x16x16_fp8_fp8 v[1:8], v[20:21], v[24:25], v[1:8]
	s_delay_alu instid0(VALU_DEP_1)
	v_wmma_f32_16x16x16_fp8_fp8 v[1:8], v[22:23], v[26:27], v[1:8]
	s_cbranch_scc1 .LBB2_19
.LBB2_12:                               ; =>This Inner Loop Header: Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 s1, s3, 0x3fffff80
	s_and_b32 s8, s4, 1
	s_wait_alu depctr_sa_sdst(0)
	v_add_nc_u32_e32 v21, s1, v15
	v_lshl_add_u32 v20, s8, 9, v16
	s_and_saveexec_b32 s9, s0
	s_cbranch_execz .LBB2_16
; %bb.13:                               ;   in Loop: Header=BB2_12 Depth=1
	v_mov_b16_e32 v12.l, 0
	v_cmp_gt_i32_e64 s1, s13, v19
	v_mov_b32_e32 v11, 0
	s_delay_alu instid0(VALU_DEP_3)
	v_mov_b16_e32 v22.l, v12.l
	v_mov_b16_e32 v24.l, v12.l
	v_mov_b16_e32 v23.l, v12.l
	s_and_b32 s1, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s14, s1
	s_cbranch_execz .LBB2_15
; %bb.14:                               ;   in Loop: Header=BB2_12 Depth=1
	v_and_b32_e32 v11, 0x78, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_nc_u32_e32 v11, v21, v11
	v_ashrrev_i32_e32 v12, 31, v11
	v_add_co_u32 v11, s1, s6, v11
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v12, null, s7, v12, s1
	global_load_b32 v12, v[11:12], off
	s_wait_loadcnt 0x0
	v_lshlrev_b32_e32 v11, 1, v12
	v_lshrrev_b32_e32 v22, 15, v12
	v_lshrrev_b32_e32 v23, 7, v12
	v_lshrrev_b32_e32 v12, 23, v12
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_and_b32_e32 v11, 0x1fe, v11
	v_and_b32_e32 v22, 0x1fe, v22
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_and_b32_e32 v24, 0x1fe, v23
	v_or_b32_e32 v25, 0x601, v12
	v_and_b32_e32 v26, 0x1fe, v12
	ds_load_u16_d16 v11, v11
	ds_load_u16_d16 v23, v22 offset:1024
	ds_load_u8_d16 v12, v25
	s_wait_dscnt 0x2
	ds_load_u16_d16_hi v11, v24 offset:512
	ds_load_u8_d16 v22, v26 offset:1536
	s_wait_dscnt 0x3
	v_lshrrev_b16 v24.l, 8, v23.l
.LBB2_15:                               ; %.loopexit
                                        ;   in Loop: Header=BB2_12 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s14
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_2)
	v_perm_b32 v23, v23, v24, 0xc0c0004
	s_wait_dscnt 0x0
	v_perm_b32 v12, v22, v12, 0xc0c0004
	v_add_nc_u32_e32 v22, v20, v18
	v_lshl_or_b32 v12, v12, 16, v23
	ds_store_b64 v22, v[11:12]
.LBB2_16:                               ; %Flow231
                                        ;   in Loop: Header=BB2_12 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s9
	s_and_saveexec_b32 s9, s0
	s_cbranch_execz .LBB2_11
; %bb.17:                               ;   in Loop: Header=BB2_12 Depth=1
	v_dual_mov_b32 v11, 0 :: v_dual_add_nc_u32 v22, 8, v19
	v_mov_b16_e32 v12.l, 0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cmp_gt_i32_e64 s1, s13, v22
	v_mov_b16_e32 v22.l, v12.l
	v_mov_b16_e32 v24.l, v12.l
	v_mov_b16_e32 v23.l, v12.l
	s_and_b32 s1, vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s14, s1
	s_cbranch_execz .LBB2_10
; %bb.18:                               ;   in Loop: Header=BB2_12 Depth=1
	v_add_nc_u32_e32 v11, 4, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_and_b32_e32 v11, 0x7c, v11
	v_add_nc_u32_e32 v11, v21, v11
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_ashrrev_i32_e32 v12, 31, v11
	v_add_co_u32 v11, s1, s6, v11
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v12, null, s7, v12, s1
	global_load_b32 v12, v[11:12], off
	s_wait_loadcnt 0x0
	v_lshlrev_b32_e32 v11, 1, v12
	v_lshrrev_b32_e32 v21, 15, v12
	v_lshrrev_b32_e32 v22, 7, v12
	v_lshrrev_b32_e32 v12, 23, v12
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_and_b32_e32 v11, 0x1fe, v11
	v_and_b32_e32 v21, 0x1fe, v21
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_and_b32_e32 v22, 0x1fe, v22
	v_or_b32_e32 v24, 0x601, v12
	v_and_b32_e32 v25, 0x1fe, v12
	ds_load_u16_d16 v11, v11
	ds_load_u16_d16 v23, v21 offset:1024
	ds_load_u8_d16 v12, v24
	s_wait_dscnt 0x2
	ds_load_u16_d16_hi v11, v22 offset:512
	ds_load_u8_d16 v22, v25 offset:1536
	s_wait_dscnt 0x3
	v_lshrrev_b16 v24.l, 8, v23.l
	s_branch .LBB2_10
.LBB2_19:                               ; %._crit_edge102.loopexit
	v_dual_mov_b32 v9, v13 :: v_dual_mov_b32 v10, v14
.LBB2_20:                               ; %Flow233
	s_delay_alu instid0(VALU_DEP_1)
	v_and_b32_e32 v0, 8, v10
	s_mul_i32 s0, ttmp7, s12
	s_wait_alu depctr_sa_sdst(0)
	s_ashr_i32 s3, s2, 31
	s_lshl_b32 s0, s0, 4
	s_wait_alu depctr_sa_sdst(0)
	s_lshl_b64 s[2:3], s[2:3], 2
	v_mad_co_u64_u32 v[9:10], null, s12, v0, v[9:10]
	v_mov_b32_e32 v10, 0
	s_ashr_i32 s1, s0, 31
	s_wait_alu depctr_sa_sdst(0)
	s_lshl_b64 s[0:1], s[0:1], 2
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[0:1], s[10:11], s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[0:1], s[0:1], s[2:3]
	v_lshlrev_b64_e32 v[11:12], 2, v[9:10]
	v_add_nc_u32_e32 v9, s12, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_4)
	v_lshlrev_b64_e32 v[13:14], 2, v[9:10]
	v_add_nc_u32_e32 v9, s12, v9
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v11, vcc_lo, s0, v11
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_add_co_ci_u32_e64 v12, null, s1, v12, vcc_lo
	v_lshlrev_b64_e32 v[15:16], 2, v[9:10]
	v_add_nc_u32_e32 v9, s12, v9
	v_add_co_u32 v13, vcc_lo, s0, v13
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v14, null, s1, v14, vcc_lo
	s_delay_alu instid0(VALU_DEP_3)
	v_lshlrev_b64_e32 v[17:18], 2, v[9:10]
	v_add_nc_u32_e32 v9, s12, v9
	s_clause 0x1
	global_store_b32 v[11:12], v1, off
	global_store_b32 v[13:14], v2, off
	v_add_co_u32 v15, vcc_lo, s0, v15
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v16, null, s1, v16, vcc_lo
	v_lshlrev_b64_e32 v[0:1], 2, v[9:10]
	v_add_nc_u32_e32 v9, s12, v9
	v_add_co_u32 v2, vcc_lo, s0, v17
	global_store_b32 v[15:16], v3, off
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v3, null, s1, v18, vcc_lo
	v_lshlrev_b64_e32 v[11:12], 2, v[9:10]
	v_add_nc_u32_e32 v9, s12, v9
	v_add_co_u32 v0, vcc_lo, s0, v0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v1, null, s1, v1, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_4) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[13:14], 2, v[9:10]
	v_add_nc_u32_e32 v9, s12, v9
	v_add_co_u32 v11, vcc_lo, s0, v11
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v12, null, s1, v12, vcc_lo
	v_lshlrev_b64_e32 v[9:10], 2, v[9:10]
	v_add_co_u32 v13, vcc_lo, s0, v13
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v14, null, s1, v14, vcc_lo
	s_delay_alu instid0(VALU_DEP_3)
	v_add_co_u32 v9, vcc_lo, s0, v9
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v10, null, s1, v10, vcc_lo
	s_clause 0x4
	global_store_b32 v[2:3], v4, off
	global_store_b32 v[0:1], v5, off
	global_store_b32 v[11:12], v6, off
	global_store_b32 v[13:14], v7, off
	global_store_b32 v[9:10], v8, off
	s_endpgm
.Lfunc_end2:
	.size	_Z12fused_m5_k32PK14__hip_fp8_e4m3PKhS3_Pfiiii, .Lfunc_end2-_Z12fused_m5_k32PK14__hip_fp8_e4m3PKhS3_Pfiiii
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel _Z12fused_m5_k32PK14__hip_fp8_e4m3PKhS3_Pfiiii
		.amdhsa_group_segment_fixed_size 3072
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 48
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_dispatch_ptr 0
		.amdhsa_user_sgpr_queue_ptr 0
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 0
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_wavefront_size32 1
		.amdhsa_uses_dynamic_stack 0
		.amdhsa_enable_private_segment 0
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 1
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 28
		.amdhsa_next_free_sgpr 15
		.amdhsa_reserve_vcc 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_fp16_overflow 0
		.amdhsa_workgroup_processor_mode 1
		.amdhsa_memory_ordered 1
		.amdhsa_forward_progress 1
		.amdhsa_inst_pref_size 14
		.amdhsa_round_robin_scheduling 0
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.text
                                        ; -- End function
	.set .L_Z12fused_m5_k32PK14__hip_fp8_e4m3PKhS3_Pfiiii.num_vgpr, 28
	.set .L_Z12fused_m5_k32PK14__hip_fp8_e4m3PKhS3_Pfiiii.num_agpr, 0
	.set .L_Z12fused_m5_k32PK14__hip_fp8_e4m3PKhS3_Pfiiii.numbered_sgpr, 15
	.set .L_Z12fused_m5_k32PK14__hip_fp8_e4m3PKhS3_Pfiiii.num_named_barrier, 0
	.set .L_Z12fused_m5_k32PK14__hip_fp8_e4m3PKhS3_Pfiiii.private_seg_size, 0
	.set .L_Z12fused_m5_k32PK14__hip_fp8_e4m3PKhS3_Pfiiii.uses_vcc, 1
	.set .L_Z12fused_m5_k32PK14__hip_fp8_e4m3PKhS3_Pfiiii.uses_flat_scratch, 0
	.set .L_Z12fused_m5_k32PK14__hip_fp8_e4m3PKhS3_Pfiiii.has_dyn_sized_stack, 0
	.set .L_Z12fused_m5_k32PK14__hip_fp8_e4m3PKhS3_Pfiiii.has_recursion, 0
	.set .L_Z12fused_m5_k32PK14__hip_fp8_e4m3PKhS3_Pfiiii.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 1700
; TotalNumSgprs: 17
; NumVgprs: 28
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 3072 bytes/workgroup (compile time only)
; SGPRBlocks: 0
; VGPRBlocks: 3
; NumSGPRsForWavesPerEU: 17
; NumVGPRsForWavesPerEU: 28
; Occupancy: 16
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 2
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
	.text
	.protected	_Z21fused_m6_repacked_k32PK14__hip_fp8_e4m3PKhS3_Pfiii ; -- Begin function _Z21fused_m6_repacked_k32PK14__hip_fp8_e4m3PKhS3_Pfiii
	.globl	_Z21fused_m6_repacked_k32PK14__hip_fp8_e4m3PKhS3_Pfiii
	.p2align	8
	.type	_Z21fused_m6_repacked_k32PK14__hip_fp8_e4m3PKhS3_Pfiii,@function
_Z21fused_m6_repacked_k32PK14__hip_fp8_e4m3PKhS3_Pfiii: ; @_Z21fused_m6_repacked_k32PK14__hip_fp8_e4m3PKhS3_Pfiii
; %bb.0:
	s_load_b256 s[4:11], s[0:1], 0x0
	v_lshlrev_b32_e32 v11, 3, v0
	s_mov_b32 s3, exec_lo
	v_cmpx_gt_u32_e32 0x100, v0
	s_cbranch_execz .LBB3_5
; %bb.1:                                ; %.lr.ph.preheader
	v_lshlrev_b32_e32 v1, 3, v0
	v_mov_b32_e32 v3, v11
	s_wait_kmcnt 0x0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v1, s2, s8, v1
	v_add_co_ci_u32_e64 v2, null, s9, 0, s2
	s_mov_b32 s8, 0
	s_branch .LBB3_3
.LBB3_2:                                ;   in Loop: Header=BB3_3 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_add_nc_u32_e32 v4, 0x100, v3
	v_cmp_lt_u32_e32 vcc_lo, 0x6ff, v3
	v_add_co_u32 v1, s2, 0x100, v1
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v2, null, 0, v2, s2
	v_mov_b32_e32 v3, v4
	s_or_b32 s8, vcc_lo, s8
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 exec_lo, exec_lo, s8
	s_cbranch_execz .LBB3_5
.LBB3_3:                                ; %.lr.ph
                                        ; =>This Inner Loop Header: Depth=1
	s_mov_b32 s2, exec_lo
	v_cmpx_gt_u32_e32 0x7f9, v3
	s_cbranch_execz .LBB3_2
; %bb.4:                                ;   in Loop: Header=BB3_3 Depth=1
	global_load_b64 v[4:5], v[1:2], off
	s_wait_loadcnt 0x0
	ds_store_b64 v3, v[4:5]
	s_branch .LBB3_2
.LBB3_5:                                ; %Flow188
	s_or_b32 exec_lo, exec_lo, s3
	s_load_b64 s[2:3], s[0:1], 0x24
	s_wait_dscnt 0x0
	s_barrier_signal -1
	v_and_b32_e32 v12, 15, v0
	v_lshrrev_b32_e32 v13, 1, v0
	s_mov_b32 s0, 0
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_kmcnt 0x0
	s_cmp_gt_i32 s3, 0
	s_cbranch_scc1 .LBB3_7
; %bb.6:                                ; %._crit_edge.._crit_edge107_crit_edge
	v_and_b32_e32 v9, 15, v0
	v_lshrrev_b32_e32 v10, 1, v0
	s_branch .LBB3_8
.LBB3_7:
	s_mov_b32 s0, -1
                                        ; implicit-def: $vgpr9
                                        ; implicit-def: $vgpr10
.LBB3_8:                                ; %Flow
	v_dual_mov_b32 v8, 0 :: v_dual_mov_b32 v7, 0
	v_dual_mov_b32 v6, 0 :: v_dual_mov_b32 v5, 0
	v_dual_mov_b32 v4, 0 :: v_dual_mov_b32 v3, 0
	v_dual_mov_b32 v2, 0 :: v_dual_mov_b32 v1, 0
	s_and_not1_b32 vcc_lo, exec_lo, s0
	s_cbranch_vccnz .LBB3_18
; %bb.9:                                ; %.lr.ph106
	v_lshlrev_b32_e32 v1, 1, v0
	v_and_b32_e32 v2, 16, v0
	s_ashr_i32 s0, s2, 31
	s_mul_i32 s1, ttmp7, s3
	s_lshr_b32 s0, s0, 28
	v_and_b32_e32 v7, 2, v1
	v_mov_b32_e32 v1, 0
	v_mad_co_u64_u32 v[3:4], null, s3, v12, v[2:3]
	v_lshl_or_b32 v4, v12, 5, 0x800
	s_add_co_i32 s8, s2, s0
	s_lshl_b32 s0, s1, 4
	v_lshl_add_u32 v5, v13, 4, 0xa00
	s_ashr_i32 s1, s0, 31
	v_dual_mov_b32 v2, v1 :: v_dual_add_nc_u32 v19, v4, v2
	v_or_b32_e32 v8, 1, v7
	s_add_nc_u64 s[0:1], s[4:5], s[0:1]
	v_lshl_add_u32 v6, v13, 5, 0x800
	v_add_co_u32 v3, s0, s0, v3
	v_dual_mov_b32 v4, v1 :: v_dual_lshlrev_b32 v15, 2, v7
	v_lshlrev_b32_e32 v7, 3, v7
	v_lshlrev_b32_e32 v17, 2, v8
	v_lshlrev_b32_e32 v8, 3, v8
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v10, null, s1, 0, s0
	v_cmp_gt_u32_e32 vcc_lo, 32, v0
	v_add_co_u32 v0, s6, s6, v11
	v_add_co_u32 v9, s0, v3, 8
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v14, null, s7, 0, s6
	v_add_co_ci_u32_e64 v10, null, 0, v10, s0
	v_add_nc_u32_e32 v15, v5, v15
	v_dual_mov_b32 v3, v1 :: v_dual_add_nc_u32 v16, v6, v7
	v_dual_mov_b32 v7, v1 :: v_dual_add_nc_u32 v18, v6, v8
	v_mov_b32_e32 v6, v1
	v_mov_b32_e32 v8, v1
	v_add_nc_u32_e32 v17, v5, v17
	v_mov_b32_e32 v5, v1
	s_wait_alu depctr_sa_sdst(0)
	s_lshl_b32 s4, s8, 4
	s_lshl_b32 s1, ttmp9, 8
	s_and_b32 s4, s4, 0xffffff00
	s_mov_b32 s5, 0
	s_branch .LBB3_11
.LBB3_10:                               ;   in Loop: Header=BB3_11 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_add_co_i32 s5, s5, 32
	s_add_co_i32 s1, s1, s4
	s_cmp_ge_i32 s5, s3
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	global_load_b128 v[20:23], v[9:10], off offset:-8
	ds_load_b128 v[24:27], v19
	v_add_co_u32 v9, s0, v9, 32
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v10, null, 0, v10, s0
	s_wait_loadcnt_dscnt 0x0
	v_wmma_f32_16x16x16_fp8_fp8 v[1:8], v[20:21], v[24:25], v[1:8]
	s_delay_alu instid0(VALU_DEP_1)
	v_wmma_f32_16x16x16_fp8_fp8 v[1:8], v[22:23], v[26:27], v[1:8]
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_cbranch_scc1 .LBB3_17
.LBB3_11:                               ; =>This Inner Loop Header: Depth=1
	s_and_saveexec_b32 s6, vcc_lo
	s_cbranch_execz .LBB3_13
; %bb.12:                               ; %.lr.ph100
                                        ;   in Loop: Header=BB3_11 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_ashr_i32 s7, s1, 31
	v_add_co_u32 v20, s0, v0, s1
	s_wait_alu depctr_sa_sdst(0) depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v21, null, s7, v14, s0
	global_load_b64 v[20:21], v[20:21], off
	s_wait_loadcnt 0x0
	ds_store_b64 v11, v[20:21] offset:2560
.LBB3_13:                               ;   in Loop: Header=BB3_11 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_and_saveexec_b32 s0, vcc_lo
	s_cbranch_execz .LBB3_15
; %bb.14:                               ;   in Loop: Header=BB3_11 Depth=1
	ds_load_b32 v20, v15
	s_wait_dscnt 0x0
	v_lshrrev_b32_e32 v21, 23, v20
	v_lshlrev_b32_e32 v23, 1, v20
	v_lshrrev_b32_e32 v22, 15, v20
	v_lshrrev_b32_e32 v20, 7, v20
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_2) | instid1(VALU_DEP_4)
	v_or_b32_e32 v24, 0x601, v21
	v_and_b32_e32 v21, 0x1fe, v21
	v_and_b32_e32 v23, 0x1fe, v23
	v_and_b32_e32 v20, 0x1fe, v20
	v_and_b32_e32 v22, 0x1fe, v22
	ds_load_u8 v24, v24
	ds_load_u8 v21, v21 offset:1536
	ds_load_u16 v23, v23
	ds_load_u16 v20, v20 offset:512
	ds_load_u16 v22, v22 offset:1024
	s_wait_dscnt 0x3
	v_perm_b32 v21, v21, v24, 0xc0c0004
	s_wait_dscnt 0x1
	v_lshl_or_b32 v20, v20, 16, v23
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_2)
	v_lshl_or_b32 v21, v21, 16, v22
	ds_store_b64 v16, v[20:21]
.LBB3_15:                               ;   in Loop: Header=BB3_11 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_saveexec_b32 s0, vcc_lo
	s_cbranch_execz .LBB3_10
; %bb.16:                               ;   in Loop: Header=BB3_11 Depth=1
	ds_load_b32 v20, v17
	s_wait_dscnt 0x0
	v_lshrrev_b32_e32 v21, 23, v20
	v_lshlrev_b32_e32 v23, 1, v20
	v_lshrrev_b32_e32 v22, 15, v20
	v_lshrrev_b32_e32 v20, 7, v20
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_2) | instid1(VALU_DEP_4)
	v_or_b32_e32 v24, 0x601, v21
	v_and_b32_e32 v21, 0x1fe, v21
	v_and_b32_e32 v23, 0x1fe, v23
	v_and_b32_e32 v20, 0x1fe, v20
	v_and_b32_e32 v22, 0x1fe, v22
	ds_load_u8 v24, v24
	ds_load_u8 v21, v21 offset:1536
	ds_load_u16 v23, v23
	ds_load_u16 v20, v20 offset:512
	ds_load_u16 v22, v22 offset:1024
	s_wait_dscnt 0x3
	v_perm_b32 v21, v21, v24, 0xc0c0004
	s_wait_dscnt 0x1
	v_lshl_or_b32 v20, v20, 16, v23
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_2)
	v_lshl_or_b32 v21, v21, 16, v22
	ds_store_b64 v18, v[20:21]
	s_branch .LBB3_10
.LBB3_17:                               ; %._crit_edge107.loopexit
	v_dual_mov_b32 v9, v12 :: v_dual_mov_b32 v10, v13
.LBB3_18:                               ; %Flow186
	s_delay_alu instid0(VALU_DEP_1)
	v_and_b32_e32 v0, 8, v10
	s_mul_i32 s0, ttmp7, s2
	s_wait_alu depctr_sa_sdst(0)
	s_lshl_b32 s4, ttmp9, 4
	s_lshl_b32 s0, s0, 4
	s_ashr_i32 s5, s4, 31
	v_mad_co_u64_u32 v[9:10], null, s2, v0, v[9:10]
	v_mov_b32_e32 v10, 0
	s_wait_alu depctr_sa_sdst(0)
	s_ashr_i32 s1, s0, 31
	s_lshl_b64 s[4:5], s[4:5], 2
	s_wait_alu depctr_sa_sdst(0)
	s_lshl_b64 s[0:1], s[0:1], 2
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[0:1], s[10:11], s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[0:1], s[0:1], s[4:5]
	v_lshlrev_b64_e32 v[11:12], 2, v[9:10]
	v_add_nc_u32_e32 v9, s2, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_4)
	v_lshlrev_b64_e32 v[13:14], 2, v[9:10]
	v_add_nc_u32_e32 v9, s2, v9
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v11, vcc_lo, s0, v11
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_add_co_ci_u32_e64 v12, null, s1, v12, vcc_lo
	v_lshlrev_b64_e32 v[15:16], 2, v[9:10]
	v_add_nc_u32_e32 v9, s2, v9
	v_add_co_u32 v13, vcc_lo, s0, v13
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v14, null, s1, v14, vcc_lo
	s_delay_alu instid0(VALU_DEP_4)
	v_add_co_u32 v15, vcc_lo, s0, v15
	v_lshlrev_b64_e32 v[17:18], 2, v[9:10]
	v_add_nc_u32_e32 v9, s2, v9
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v16, null, s1, v16, vcc_lo
	s_clause 0x2
	global_store_b32 v[11:12], v1, off
	global_store_b32 v[13:14], v2, off
	global_store_b32 v[15:16], v3, off
	v_lshlrev_b64_e32 v[0:1], 2, v[9:10]
	v_add_nc_u32_e32 v9, s2, v9
	v_add_co_u32 v2, vcc_lo, s0, v17
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v3, null, s1, v18, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_4) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[11:12], 2, v[9:10]
	v_add_nc_u32_e32 v9, s2, v9
	v_add_co_u32 v0, vcc_lo, s0, v0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v1, null, s1, v1, vcc_lo
	v_lshlrev_b64_e32 v[13:14], 2, v[9:10]
	v_add_nc_u32_e32 v9, s2, v9
	v_add_co_u32 v11, vcc_lo, s0, v11
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v12, null, s1, v12, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_3) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[9:10], 2, v[9:10]
	v_add_co_u32 v13, vcc_lo, s0, v13
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v14, null, s1, v14, vcc_lo
	v_add_co_u32 v9, vcc_lo, s0, v9
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v10, null, s1, v10, vcc_lo
	s_clause 0x4
	global_store_b32 v[2:3], v4, off
	global_store_b32 v[0:1], v5, off
	global_store_b32 v[11:12], v6, off
	global_store_b32 v[13:14], v7, off
	global_store_b32 v[9:10], v8, off
	s_endpgm
.Lfunc_end3:
	.size	_Z21fused_m6_repacked_k32PK14__hip_fp8_e4m3PKhS3_Pfiii, .Lfunc_end3-_Z21fused_m6_repacked_k32PK14__hip_fp8_e4m3PKhS3_Pfiii
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel _Z21fused_m6_repacked_k32PK14__hip_fp8_e4m3PKhS3_Pfiii
		.amdhsa_group_segment_fixed_size 2816
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 44
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_dispatch_ptr 0
		.amdhsa_user_sgpr_queue_ptr 0
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 0
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_wavefront_size32 1
		.amdhsa_uses_dynamic_stack 0
		.amdhsa_enable_private_segment 0
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 1
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 28
		.amdhsa_next_free_sgpr 12
		.amdhsa_reserve_vcc 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_fp16_overflow 0
		.amdhsa_workgroup_processor_mode 1
		.amdhsa_memory_ordered 1
		.amdhsa_forward_progress 1
		.amdhsa_inst_pref_size 13
		.amdhsa_round_robin_scheduling 0
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.text
                                        ; -- End function
	.set .L_Z21fused_m6_repacked_k32PK14__hip_fp8_e4m3PKhS3_Pfiii.num_vgpr, 28
	.set .L_Z21fused_m6_repacked_k32PK14__hip_fp8_e4m3PKhS3_Pfiii.num_agpr, 0
	.set .L_Z21fused_m6_repacked_k32PK14__hip_fp8_e4m3PKhS3_Pfiii.numbered_sgpr, 12
	.set .L_Z21fused_m6_repacked_k32PK14__hip_fp8_e4m3PKhS3_Pfiii.num_named_barrier, 0
	.set .L_Z21fused_m6_repacked_k32PK14__hip_fp8_e4m3PKhS3_Pfiii.private_seg_size, 0
	.set .L_Z21fused_m6_repacked_k32PK14__hip_fp8_e4m3PKhS3_Pfiii.uses_vcc, 1
	.set .L_Z21fused_m6_repacked_k32PK14__hip_fp8_e4m3PKhS3_Pfiii.uses_flat_scratch, 0
	.set .L_Z21fused_m6_repacked_k32PK14__hip_fp8_e4m3PKhS3_Pfiii.has_dyn_sized_stack, 0
	.set .L_Z21fused_m6_repacked_k32PK14__hip_fp8_e4m3PKhS3_Pfiii.has_recursion, 0
	.set .L_Z21fused_m6_repacked_k32PK14__hip_fp8_e4m3PKhS3_Pfiii.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 1552
; TotalNumSgprs: 14
; NumVgprs: 28
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 2816 bytes/workgroup (compile time only)
; SGPRBlocks: 0
; VGPRBlocks: 3
; NumSGPRsForWavesPerEU: 14
; NumVGPRsForWavesPerEU: 28
; Occupancy: 16
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 2
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
	.text
	.protected	_Z24fused_m6_repacked_db_k32PK14__hip_fp8_e4m3PKhS3_Pfiii ; -- Begin function _Z24fused_m6_repacked_db_k32PK14__hip_fp8_e4m3PKhS3_Pfiii
	.globl	_Z24fused_m6_repacked_db_k32PK14__hip_fp8_e4m3PKhS3_Pfiii
	.p2align	8
	.type	_Z24fused_m6_repacked_db_k32PK14__hip_fp8_e4m3PKhS3_Pfiii,@function
_Z24fused_m6_repacked_db_k32PK14__hip_fp8_e4m3PKhS3_Pfiii: ; @_Z24fused_m6_repacked_db_k32PK14__hip_fp8_e4m3PKhS3_Pfiii
; %bb.0:
	s_load_b256 s[4:11], s[0:1], 0x0
	v_lshlrev_b32_e32 v10, 3, v0
	s_mov_b32 s3, exec_lo
	v_cmpx_gt_u32_e32 0x100, v0
	s_cbranch_execz .LBB4_5
; %bb.1:                                ; %.lr.ph.preheader
	v_lshlrev_b32_e32 v1, 3, v0
	v_mov_b32_e32 v3, v10
	s_wait_kmcnt 0x0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v1, s2, s8, v1
	v_add_co_ci_u32_e64 v2, null, s9, 0, s2
	s_mov_b32 s8, 0
	s_branch .LBB4_3
.LBB4_2:                                ;   in Loop: Header=BB4_3 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_add_nc_u32_e32 v4, 0x100, v3
	v_cmp_lt_u32_e32 vcc_lo, 0x6ff, v3
	v_add_co_u32 v1, s2, 0x100, v1
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v2, null, 0, v2, s2
	v_mov_b32_e32 v3, v4
	s_or_b32 s8, vcc_lo, s8
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 exec_lo, exec_lo, s8
	s_cbranch_execz .LBB4_5
.LBB4_3:                                ; %.lr.ph
                                        ; =>This Inner Loop Header: Depth=1
	s_mov_b32 s2, exec_lo
	v_cmpx_gt_u32_e32 0x7f9, v3
	s_cbranch_execz .LBB4_2
; %bb.4:                                ;   in Loop: Header=BB4_3 Depth=1
	global_load_b64 v[4:5], v[1:2], off
	s_wait_loadcnt 0x0
	ds_store_b64 v3, v[4:5]
	s_branch .LBB4_2
.LBB4_5:                                ; %Flow210
	s_or_b32 exec_lo, exec_lo, s3
	s_load_b64 s[2:3], s[0:1], 0x24
	s_wait_dscnt 0x0
	s_barrier_signal -1
	v_cmp_gt_u32_e64 s0, 32, v0
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_kmcnt 0x0
	s_cmp_gt_i32 s3, 31
	s_cselect_b32 s1, -1, 0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_b32 s8, s1, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s1, s8
	s_cbranch_execz .LBB4_7
; %bb.6:                                ; %.lr.ph124
	s_lshl_b32 s8, ttmp9, 8
	s_wait_alu depctr_sa_sdst(0)
	s_ashr_i32 s9, s8, 31
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[8:9], s[6:7], s[8:9]
	global_load_b64 v[1:2], v10, s[8:9]
	s_wait_loadcnt 0x0
	ds_store_b64 v10, v[1:2] offset:3072
.LBB4_7:                                ; %.loopexit
	s_or_b32 exec_lo, exec_lo, s1
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_and_b32_e32 v11, 15, v0
	v_lshrrev_b32_e32 v12, 1, v0
	s_mov_b32 s1, 0
	s_cmp_gt_i32 s3, 0
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_cbranch_scc1 .LBB4_9
; %bb.8:                                ; %.loopexit.._crit_edge134_crit_edge
	v_and_b32_e32 v9, 15, v0
	v_lshrrev_b32_e32 v13, 1, v0
	s_branch .LBB4_10
.LBB4_9:
	s_mov_b32 s1, -1
                                        ; implicit-def: $vgpr9
                                        ; implicit-def: $vgpr13
.LBB4_10:                               ; %Flow207
	v_dual_mov_b32 v8, 0 :: v_dual_mov_b32 v7, 0
	v_dual_mov_b32 v6, 0 :: v_dual_mov_b32 v5, 0
	v_dual_mov_b32 v4, 0 :: v_dual_mov_b32 v3, 0
	v_dual_mov_b32 v2, 0 :: v_dual_mov_b32 v1, 0
	s_and_not1_b32 vcc_lo, exec_lo, s1
	s_cbranch_vccnz .LBB4_22
; %bb.11:                               ; %.lr.ph133
	v_lshlrev_b32_e32 v1, 1, v0
	v_and_b32_e32 v0, 16, v0
	s_mul_i32 s8, ttmp7, s3
	s_ashr_i32 s1, s2, 31
	s_wait_alu depctr_sa_sdst(0)
	s_lshl_b32 s8, s8, 4
	v_and_b32_e32 v4, 2, v1
	v_mad_co_u64_u32 v[2:3], null, s3, v11, v[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_ashr_i32 s9, s8, 31
	s_lshr_b32 s1, s1, 28
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[4:5], s[4:5], s[8:9]
	v_or_b32_e32 v3, 1, v4
	v_mov_b32_e32 v1, 0
	v_lshlrev_b32_e32 v5, 5, v11
	s_add_co_i32 s1, s2, s1
	v_add_co_u32 v2, s4, s4, v2
	s_delay_alu instid0(VALU_DEP_3)
	v_dual_mov_b32 v6, v1 :: v_dual_lshlrev_b32 v17, 2, v3
	v_dual_mov_b32 v7, v1 :: v_dual_lshlrev_b32 v18, 3, v3
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v3, null, s5, 0, s4
	s_ashr_i32 s12, s1, 4
	v_add_co_u32 v19, s1, s6, v10
	v_add_co_u32 v9, vcc_lo, v2, 8
	v_lshl_add_u32 v13, v12, 4, 0xc00
	v_lshl_add_u32 v14, v12, 5, 0x800
	v_add_co_ci_u32_e64 v20, null, s7, 0, s1
	v_dual_mov_b32 v8, v1 :: v_dual_add_nc_u32 v21, 0xc00, v10
	v_add_co_ci_u32_e64 v10, null, 0, v3, vcc_lo
	v_mov_b32_e32 v2, v1
	v_or3_b32 v0, v5, v0, 0x800
	v_mov_b32_e32 v3, v1
	v_lshlrev_b32_e32 v15, 2, v4
	v_dual_mov_b32 v5, v1 :: v_dual_lshlrev_b32 v16, 3, v4
	v_mov_b32_e32 v4, v1
	s_add_co_i32 s1, ttmp9, s12
	s_lshl_b32 s4, s12, 8
	s_wait_alu depctr_sa_sdst(0)
	s_lshl_b32 s1, s1, 8
	s_mov_b32 s5, 32
	s_mov_b32 s6, 0
	s_branch .LBB4_14
.LBB4_12:                               ;   in Loop: Header=BB4_14 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s8
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_add_co_u32 v9, vcc_lo, v9, 32
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v10, null, 0, v10, vcc_lo
	s_add_co_i32 s1, s1, s4
	s_add_co_i32 s5, s5, 32
	s_add_co_i32 s6, s6, 1
	s_mov_b32 s8, 0
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
.LBB4_13:                               ; %Flow
                                        ;   in Loop: Header=BB4_14 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 vcc_lo, exec_lo, s8
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB4_21
.LBB4_14:                               ; =>This Inner Loop Header: Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 s7, s6, 1
	s_wait_alu depctr_sa_sdst(0)
	v_lshl_add_u32 v23, s7, 8, v13
	v_lshl_add_u32 v22, s7, 9, v14
	s_and_saveexec_b32 s8, s0
	s_cbranch_execz .LBB4_16
; %bb.15:                               ;   in Loop: Header=BB4_14 Depth=1
	s_delay_alu instid0(VALU_DEP_2)
	v_add_nc_u32_e32 v24, v23, v15
	ds_load_b32 v24, v24
	s_wait_dscnt 0x0
	v_lshrrev_b32_e32 v25, 23, v24
	v_lshlrev_b32_e32 v26, 1, v24
	v_lshrrev_b32_e32 v27, 7, v24
	v_lshrrev_b32_e32 v24, 15, v24
	s_delay_alu instid0(VALU_DEP_4)
	v_or_b32_e32 v28, 0x601, v25
	v_and_b32_e32 v25, 0x1fe, v25
	v_and_b32_e32 v26, 0x1fe, v26
	v_and_b32_e32 v27, 0x1fe, v27
	v_and_b32_e32 v24, 0x1fe, v24
	ds_load_u8 v28, v28
	ds_load_u8 v25, v25 offset:1536
	ds_load_u16 v26, v26
	ds_load_u16 v27, v27 offset:512
	ds_load_u16 v29, v24 offset:1024
	s_wait_dscnt 0x3
	v_perm_b32 v25, v25, v28, 0xc0c0004
	v_add_nc_u32_e32 v28, v22, v16
	s_wait_dscnt 0x1
	v_lshl_or_b32 v24, v27, 16, v26
	s_wait_dscnt 0x0
	v_lshl_or_b32 v25, v25, 16, v29
	ds_store_b64 v28, v[24:25]
.LBB4_16:                               ;   in Loop: Header=BB4_14 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s8
	s_and_saveexec_b32 s8, s0
	s_cbranch_execz .LBB4_18
; %bb.17:                               ;   in Loop: Header=BB4_14 Depth=1
	v_add_nc_u32_e32 v23, v23, v17
	ds_load_b32 v23, v23
	s_wait_dscnt 0x0
	v_lshrrev_b32_e32 v24, 23, v23
	v_lshlrev_b32_e32 v25, 1, v23
	v_lshrrev_b32_e32 v26, 7, v23
	v_lshrrev_b32_e32 v23, 15, v23
	s_delay_alu instid0(VALU_DEP_4)
	v_or_b32_e32 v27, 0x601, v24
	v_and_b32_e32 v24, 0x1fe, v24
	v_and_b32_e32 v25, 0x1fe, v25
	v_and_b32_e32 v26, 0x1fe, v26
	v_and_b32_e32 v23, 0x1fe, v23
	ds_load_u8 v27, v27
	ds_load_u8 v24, v24 offset:1536
	ds_load_u16 v25, v25
	ds_load_u16 v26, v26 offset:512
	ds_load_u16 v23, v23 offset:1024
	s_wait_dscnt 0x3
	v_perm_b32 v24, v24, v27, 0xc0c0004
	v_add_nc_u32_e32 v27, v22, v18
	s_wait_dscnt 0x1
	v_lshl_or_b32 v22, v26, 16, v25
	s_wait_dscnt 0x0
	v_lshl_or_b32 v23, v24, 16, v23
	ds_store_b64 v27, v[22:23]
.LBB4_18:                               ;   in Loop: Header=BB4_14 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s8
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_lshl_add_u32 v26, s7, 9, v0
	s_cmp_ge_i32 s5, s3
	s_mov_b32 s8, -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	global_load_b128 v[22:25], v[9:10], off offset:-8
	ds_load_b128 v[26:29], v26
	s_wait_loadcnt_dscnt 0x0
	v_wmma_f32_16x16x16_fp8_fp8 v[1:8], v[22:23], v[26:27], v[1:8]
	s_delay_alu instid0(VALU_DEP_1)
	v_wmma_f32_16x16x16_fp8_fp8 v[1:8], v[24:25], v[28:29], v[1:8]
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_cbranch_scc1 .LBB4_13
; %bb.19:                               ;   in Loop: Header=BB4_14 Depth=1
	s_and_saveexec_b32 s8, s0
	s_cbranch_execz .LBB4_12
; %bb.20:                               ; %.lr.ph128
                                        ;   in Loop: Header=BB4_14 Depth=1
	s_ashr_i32 s9, s1, 31
	v_add_co_u32 v22, vcc_lo, v19, s1
	s_wait_alu depctr_sa_sdst(0) depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v23, null, s9, v20, vcc_lo
	s_xor_b32 s7, s7, 1
	s_wait_alu depctr_sa_sdst(0)
	v_lshl_add_u32 v24, s7, 8, v21
	global_load_b64 v[22:23], v[22:23], off
	s_wait_loadcnt 0x0
	ds_store_b64 v24, v[22:23]
	s_branch .LBB4_12
.LBB4_21:                               ; %._crit_edge134.loopexit
	v_mov_b32_e32 v9, v11
	v_mov_b32_e32 v13, v12
.LBB4_22:                               ; %Flow208
	s_delay_alu instid0(VALU_DEP_1)
	v_and_b32_e32 v0, 8, v13
	s_mul_i32 s0, ttmp7, s2
	s_wait_alu depctr_sa_sdst(0)
	s_lshl_b32 s4, ttmp9, 4
	s_lshl_b32 s0, s0, 4
	s_wait_alu depctr_sa_sdst(0)
	s_ashr_i32 s5, s4, 31
	v_mad_co_u64_u32 v[9:10], null, s2, v0, v[9:10]
	v_mov_b32_e32 v10, 0
	s_ashr_i32 s1, s0, 31
	s_wait_alu depctr_sa_sdst(0)
	s_lshl_b64 s[4:5], s[4:5], 2
	s_lshl_b64 s[0:1], s[0:1], 2
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[0:1], s[10:11], s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[0:1], s[0:1], s[4:5]
	v_lshlrev_b64_e32 v[11:12], 2, v[9:10]
	v_add_nc_u32_e32 v9, s2, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_4)
	v_lshlrev_b64_e32 v[13:14], 2, v[9:10]
	v_add_nc_u32_e32 v9, s2, v9
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v11, vcc_lo, s0, v11
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v12, null, s1, v12, vcc_lo
	v_lshlrev_b64_e32 v[15:16], 2, v[9:10]
	v_add_nc_u32_e32 v9, s2, v9
	v_add_co_u32 v13, vcc_lo, s0, v13
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v14, null, s1, v14, vcc_lo
	s_delay_alu instid0(VALU_DEP_4)
	v_add_co_u32 v15, vcc_lo, s0, v15
	v_lshlrev_b64_e32 v[17:18], 2, v[9:10]
	v_add_nc_u32_e32 v9, s2, v9
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v16, null, s1, v16, vcc_lo
	s_clause 0x2
	global_store_b32 v[11:12], v1, off
	global_store_b32 v[13:14], v2, off
	global_store_b32 v[15:16], v3, off
	v_lshlrev_b64_e32 v[0:1], 2, v[9:10]
	v_add_nc_u32_e32 v9, s2, v9
	v_add_co_u32 v2, vcc_lo, s0, v17
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v3, null, s1, v18, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_4) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[11:12], 2, v[9:10]
	v_add_nc_u32_e32 v9, s2, v9
	v_add_co_u32 v0, vcc_lo, s0, v0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v1, null, s1, v1, vcc_lo
	v_lshlrev_b64_e32 v[13:14], 2, v[9:10]
	v_add_nc_u32_e32 v9, s2, v9
	v_add_co_u32 v11, vcc_lo, s0, v11
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v12, null, s1, v12, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_3) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[9:10], 2, v[9:10]
	v_add_co_u32 v13, vcc_lo, s0, v13
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v14, null, s1, v14, vcc_lo
	v_add_co_u32 v9, vcc_lo, s0, v9
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v10, null, s1, v10, vcc_lo
	s_clause 0x4
	global_store_b32 v[2:3], v4, off
	global_store_b32 v[0:1], v5, off
	global_store_b32 v[11:12], v6, off
	global_store_b32 v[13:14], v7, off
	global_store_b32 v[9:10], v8, off
	s_endpgm
.Lfunc_end4:
	.size	_Z24fused_m6_repacked_db_k32PK14__hip_fp8_e4m3PKhS3_Pfiii, .Lfunc_end4-_Z24fused_m6_repacked_db_k32PK14__hip_fp8_e4m3PKhS3_Pfiii
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel _Z24fused_m6_repacked_db_k32PK14__hip_fp8_e4m3PKhS3_Pfiii
		.amdhsa_group_segment_fixed_size 3584
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 44
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_dispatch_ptr 0
		.amdhsa_user_sgpr_queue_ptr 0
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 0
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_wavefront_size32 1
		.amdhsa_uses_dynamic_stack 0
		.amdhsa_enable_private_segment 0
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 1
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 30
		.amdhsa_next_free_sgpr 13
		.amdhsa_reserve_vcc 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_fp16_overflow 0
		.amdhsa_workgroup_processor_mode 1
		.amdhsa_memory_ordered 1
		.amdhsa_forward_progress 1
		.amdhsa_inst_pref_size 14
		.amdhsa_round_robin_scheduling 0
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.text
                                        ; -- End function
	.set .L_Z24fused_m6_repacked_db_k32PK14__hip_fp8_e4m3PKhS3_Pfiii.num_vgpr, 30
	.set .L_Z24fused_m6_repacked_db_k32PK14__hip_fp8_e4m3PKhS3_Pfiii.num_agpr, 0
	.set .L_Z24fused_m6_repacked_db_k32PK14__hip_fp8_e4m3PKhS3_Pfiii.numbered_sgpr, 13
	.set .L_Z24fused_m6_repacked_db_k32PK14__hip_fp8_e4m3PKhS3_Pfiii.num_named_barrier, 0
	.set .L_Z24fused_m6_repacked_db_k32PK14__hip_fp8_e4m3PKhS3_Pfiii.private_seg_size, 0
	.set .L_Z24fused_m6_repacked_db_k32PK14__hip_fp8_e4m3PKhS3_Pfiii.uses_vcc, 1
	.set .L_Z24fused_m6_repacked_db_k32PK14__hip_fp8_e4m3PKhS3_Pfiii.uses_flat_scratch, 0
	.set .L_Z24fused_m6_repacked_db_k32PK14__hip_fp8_e4m3PKhS3_Pfiii.has_dyn_sized_stack, 0
	.set .L_Z24fused_m6_repacked_db_k32PK14__hip_fp8_e4m3PKhS3_Pfiii.has_recursion, 0
	.set .L_Z24fused_m6_repacked_db_k32PK14__hip_fp8_e4m3PKhS3_Pfiii.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 1752
; TotalNumSgprs: 15
; NumVgprs: 30
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 3584 bytes/workgroup (compile time only)
; SGPRBlocks: 0
; VGPRBlocks: 3
; NumSGPRsForWavesPerEU: 15
; NumVGPRsForWavesPerEU: 30
; Occupancy: 16
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 2
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
	.text
	.protected	_Z21fused_m6_repacked_k64PK14__hip_fp8_e4m3PKhS3_Pfiii ; -- Begin function _Z21fused_m6_repacked_k64PK14__hip_fp8_e4m3PKhS3_Pfiii
	.globl	_Z21fused_m6_repacked_k64PK14__hip_fp8_e4m3PKhS3_Pfiii
	.p2align	8
	.type	_Z21fused_m6_repacked_k64PK14__hip_fp8_e4m3PKhS3_Pfiii,@function
_Z21fused_m6_repacked_k64PK14__hip_fp8_e4m3PKhS3_Pfiii: ; @_Z21fused_m6_repacked_k64PK14__hip_fp8_e4m3PKhS3_Pfiii
; %bb.0:
	s_load_b256 s[4:11], s[0:1], 0x0
	v_lshlrev_b32_e32 v10, 3, v0
	s_mov_b32 s3, exec_lo
	v_cmpx_gt_u32_e32 0x100, v0
	s_cbranch_execz .LBB5_5
; %bb.1:                                ; %.lr.ph.preheader
	v_lshlrev_b32_e32 v1, 3, v0
	v_mov_b32_e32 v3, v10
	s_wait_kmcnt 0x0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v1, s2, s8, v1
	v_add_co_ci_u32_e64 v2, null, s9, 0, s2
	s_mov_b32 s8, 0
	s_branch .LBB5_3
.LBB5_2:                                ;   in Loop: Header=BB5_3 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_add_nc_u32_e32 v4, 0x100, v3
	v_cmp_lt_u32_e32 vcc_lo, 0x6ff, v3
	v_add_co_u32 v1, s2, 0x100, v1
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v2, null, 0, v2, s2
	v_mov_b32_e32 v3, v4
	s_or_b32 s8, vcc_lo, s8
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 exec_lo, exec_lo, s8
	s_cbranch_execz .LBB5_5
.LBB5_3:                                ; %.lr.ph
                                        ; =>This Inner Loop Header: Depth=1
	s_mov_b32 s2, exec_lo
	v_cmpx_gt_u32_e32 0x7f9, v3
	s_cbranch_execz .LBB5_2
; %bb.4:                                ;   in Loop: Header=BB5_3 Depth=1
	global_load_b64 v[4:5], v[1:2], off
	s_wait_loadcnt 0x0
	ds_store_b64 v3, v[4:5]
	s_branch .LBB5_2
.LBB5_5:                                ; %Flow209
	s_or_b32 exec_lo, exec_lo, s3
	s_load_b64 s[2:3], s[0:1], 0x24
	s_wait_dscnt 0x0
	s_barrier_signal -1
	v_and_b32_e32 v11, 15, v0
	v_lshrrev_b32_e32 v12, 1, v0
	s_mov_b32 s0, 0
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_kmcnt 0x0
	s_cmp_gt_i32 s3, 0
	s_cbranch_scc1 .LBB5_7
; %bb.6:                                ; %._crit_edge.._crit_edge100_crit_edge
	v_and_b32_e32 v9, 15, v0
	v_lshrrev_b32_e32 v13, 1, v0
	s_branch .LBB5_8
.LBB5_7:
	s_mov_b32 s0, -1
                                        ; implicit-def: $vgpr9
                                        ; implicit-def: $vgpr13
.LBB5_8:                                ; %Flow206
	v_dual_mov_b32 v8, 0 :: v_dual_mov_b32 v7, 0
	v_dual_mov_b32 v6, 0 :: v_dual_mov_b32 v5, 0
	v_dual_mov_b32 v4, 0 :: v_dual_mov_b32 v3, 0
	v_dual_mov_b32 v2, 0 :: v_dual_mov_b32 v1, 0
	s_and_not1_b32 vcc_lo, exec_lo, s0
	s_cbranch_vccnz .LBB5_23
; %bb.9:                                ; %.lr.ph99
	v_lshlrev_b32_e32 v1, 1, v0
	s_ashr_i32 s0, s2, 31
	s_mul_i32 s1, ttmp7, s3
	s_lshr_b32 s0, s0, 28
	v_cmp_gt_u32_e32 vcc_lo, 64, v0
	v_dual_mov_b32 v1, 0 :: v_dual_and_b32 v2, 32, v1
	s_add_co_i32 s8, s2, s0
	s_lshl_b32 s0, s1, 4
	v_lshlrev_b32_e32 v7, 6, v0
	s_delay_alu instid0(VALU_DEP_2)
	v_mad_co_u64_u32 v[3:4], null, s3, v11, v[2:3]
	s_ashr_i32 s1, s0, 31
	v_lshlrev_b32_e32 v5, 2, v0
	s_add_nc_u64 s[0:1], s[4:5], s[0:1]
	s_movk_i32 s4, 0x3c0
	v_lshl_add_u32 v6, v12, 5, 0xc00
	v_lshl_add_u32 v4, v12, 6, 0x800
	v_add_co_u32 v13, s0, s0, v3
	v_and_or_b32 v3, v7, s4, 0x800
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v14, null, s1, 0, s0
	v_cmp_gt_u32_e64 s0, 32, v0
	v_add_co_u32 v15, s1, s6, v10
	v_dual_mov_b32 v2, v1 :: v_dual_add_nc_u32 v25, v3, v2
	v_and_b32_e32 v5, 4, v5
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v16, null, s7, 0, s1
	s_wait_alu depctr_sa_sdst(0)
	s_lshl_b32 s1, s8, 5
	s_lshl_b32 s4, ttmp9, 9
	v_or_b32_e32 v0, 1, v5
	v_lshlrev_b32_e32 v7, 2, v5
	v_or_b32_e32 v8, 2, v5
	v_lshlrev_b32_e32 v9, 3, v5
	v_or_b32_e32 v5, 3, v5
	v_lshlrev_b32_e32 v19, 2, v0
	v_dual_mov_b32 v3, v1 :: v_dual_lshlrev_b32 v20, 3, v0
	v_lshlrev_b32_e32 v21, 2, v8
	v_lshlrev_b32_e32 v8, 3, v8
	v_lshlrev_b32_e32 v23, 2, v5
	v_lshlrev_b32_e32 v5, 3, v5
	v_add_nc_u32_e32 v0, 0xffffff00, v10
	v_add_nc_u32_e32 v18, v4, v9
	v_add_nc_u32_e32 v19, v6, v19
	v_add_nc_u32_e32 v20, v4, v20
	v_add_nc_u32_e32 v21, v6, v21
	v_add_nc_u32_e32 v23, v6, v23
	v_dual_mov_b32 v5, v1 :: v_dual_add_nc_u32 v24, v4, v5
	v_add_nc_u32_e32 v22, v4, v8
	v_mov_b32_e32 v4, v1
	v_dual_mov_b32 v8, v1 :: v_dual_add_nc_u32 v17, v6, v7
	v_dual_mov_b32 v6, v1 :: v_dual_mov_b32 v7, v1
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 s5, s1, 0xfffffe00
	s_mov_b32 s6, 0
	s_branch .LBB5_11
.LBB5_10:                               ;   in Loop: Header=BB5_11 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s1
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_add_co_u32 v9, s1, v13, s6
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v10, null, 0, v14, s1
	s_add_co_i32 s6, s6, 64
	s_add_co_i32 s4, s4, s5
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_ge_i32 s6, s3
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_clause 0x1
	global_load_b128 v[26:29], v[9:10], off offset:16
	global_load_b128 v[30:33], v[9:10], off
	ds_load_b128 v[34:37], v25
	ds_load_b128 v[38:41], v25 offset:16
	s_wait_loadcnt_dscnt 0x1
	v_wmma_f32_16x16x16_fp8_fp8 v[1:8], v[30:31], v[34:35], v[1:8]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_wmma_f32_16x16x16_fp8_fp8 v[1:8], v[32:33], v[36:37], v[1:8]
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_fp8_fp8 v[1:8], v[26:27], v[38:39], v[1:8]
	s_delay_alu instid0(VALU_DEP_1)
	v_wmma_f32_16x16x16_fp8_fp8 v[1:8], v[28:29], v[40:41], v[1:8]
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_cbranch_scc1 .LBB5_22
.LBB5_11:                               ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB5_13 Depth 2
	s_and_saveexec_b32 s7, vcc_lo
	s_cbranch_execz .LBB5_14
; %bb.12:                               ; %.lr.ph93
                                        ;   in Loop: Header=BB5_11 Depth=1
	s_ashr_i32 s8, s4, 31
	v_add_co_u32 v9, s1, v15, s4
	s_wait_alu depctr_sa_sdst(0) depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v10, null, s8, v16, s1
	v_mov_b32_e32 v26, v0
	s_mov_b32 s8, 0
.LBB5_13:                               ;   Parent Loop BB5_11 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	global_load_b64 v[27:28], v[9:10], off
	v_add_co_u32 v9, s1, 0x100, v9
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v10, null, 0, v10, s1
	s_wait_loadcnt 0x0
	ds_store_b64 v26, v[27:28] offset:3328
	v_add_co_u32 v26, s9, 0x100, v26
	s_xor_b32 s9, s9, -1
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 s1, exec_lo, s9
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s8, s1, s8
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 exec_lo, exec_lo, s8
	s_cbranch_execnz .LBB5_13
.LBB5_14:                               ; %Flow205
                                        ;   in Loop: Header=BB5_11 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s7
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_and_saveexec_b32 s1, s0
	s_cbranch_execz .LBB5_16
; %bb.15:                               ;   in Loop: Header=BB5_11 Depth=1
	ds_load_b32 v9, v17
	s_wait_dscnt 0x0
	v_lshrrev_b32_e32 v10, 23, v9
	v_lshlrev_b32_e32 v26, 1, v9
	v_lshrrev_b32_e32 v27, 7, v9
	v_lshrrev_b32_e32 v9, 15, v9
	s_delay_alu instid0(VALU_DEP_4)
	v_or_b32_e32 v28, 0x601, v10
	v_and_b32_e32 v10, 0x1fe, v10
	v_and_b32_e32 v26, 0x1fe, v26
	v_and_b32_e32 v27, 0x1fe, v27
	v_and_b32_e32 v9, 0x1fe, v9
	ds_load_u8 v28, v28
	ds_load_u8 v10, v10 offset:1536
	ds_load_u16 v26, v26
	ds_load_u16 v27, v27 offset:512
	ds_load_u16 v29, v9 offset:1024
	s_wait_dscnt 0x3
	v_perm_b32 v10, v10, v28, 0xc0c0004
	s_wait_dscnt 0x1
	v_lshl_or_b32 v9, v27, 16, v26
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_2)
	v_lshl_or_b32 v10, v10, 16, v29
	ds_store_b64 v18, v[9:10]
.LBB5_16:                               ;   in Loop: Header=BB5_11 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s1
	s_and_saveexec_b32 s1, s0
	s_cbranch_execz .LBB5_18
; %bb.17:                               ;   in Loop: Header=BB5_11 Depth=1
	ds_load_b32 v9, v19
	s_wait_dscnt 0x0
	v_lshrrev_b32_e32 v10, 23, v9
	v_lshlrev_b32_e32 v26, 1, v9
	v_lshrrev_b32_e32 v27, 7, v9
	v_lshrrev_b32_e32 v9, 15, v9
	s_delay_alu instid0(VALU_DEP_4)
	v_or_b32_e32 v28, 0x601, v10
	v_and_b32_e32 v10, 0x1fe, v10
	v_and_b32_e32 v26, 0x1fe, v26
	v_and_b32_e32 v27, 0x1fe, v27
	v_and_b32_e32 v9, 0x1fe, v9
	ds_load_u8 v28, v28
	ds_load_u8 v10, v10 offset:1536
	ds_load_u16 v26, v26
	ds_load_u16 v27, v27 offset:512
	ds_load_u16 v29, v9 offset:1024
	s_wait_dscnt 0x3
	v_perm_b32 v10, v10, v28, 0xc0c0004
	s_wait_dscnt 0x1
	v_lshl_or_b32 v9, v27, 16, v26
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_2)
	v_lshl_or_b32 v10, v10, 16, v29
	ds_store_b64 v20, v[9:10]
.LBB5_18:                               ;   in Loop: Header=BB5_11 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s1
	s_and_saveexec_b32 s1, s0
	s_cbranch_execz .LBB5_20
; %bb.19:                               ;   in Loop: Header=BB5_11 Depth=1
	ds_load_b32 v9, v21
	s_wait_dscnt 0x0
	v_lshrrev_b32_e32 v10, 23, v9
	v_lshlrev_b32_e32 v26, 1, v9
	v_lshrrev_b32_e32 v27, 7, v9
	v_lshrrev_b32_e32 v9, 15, v9
	s_delay_alu instid0(VALU_DEP_4)
	v_or_b32_e32 v28, 0x601, v10
	v_and_b32_e32 v10, 0x1fe, v10
	v_and_b32_e32 v26, 0x1fe, v26
	v_and_b32_e32 v27, 0x1fe, v27
	v_and_b32_e32 v9, 0x1fe, v9
	ds_load_u8 v28, v28
	ds_load_u8 v10, v10 offset:1536
	ds_load_u16 v26, v26
	ds_load_u16 v27, v27 offset:512
	ds_load_u16 v29, v9 offset:1024
	s_wait_dscnt 0x3
	v_perm_b32 v10, v10, v28, 0xc0c0004
	s_wait_dscnt 0x1
	v_lshl_or_b32 v9, v27, 16, v26
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_2)
	v_lshl_or_b32 v10, v10, 16, v29
	ds_store_b64 v22, v[9:10]
.LBB5_20:                               ;   in Loop: Header=BB5_11 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s1
	s_and_saveexec_b32 s1, s0
	s_cbranch_execz .LBB5_10
; %bb.21:                               ;   in Loop: Header=BB5_11 Depth=1
	ds_load_b32 v9, v23
	s_wait_dscnt 0x0
	v_lshrrev_b32_e32 v10, 23, v9
	v_lshlrev_b32_e32 v26, 1, v9
	v_lshrrev_b32_e32 v27, 7, v9
	v_lshrrev_b32_e32 v9, 15, v9
	s_delay_alu instid0(VALU_DEP_4)
	v_or_b32_e32 v28, 0x601, v10
	v_and_b32_e32 v10, 0x1fe, v10
	v_and_b32_e32 v26, 0x1fe, v26
	v_and_b32_e32 v27, 0x1fe, v27
	v_and_b32_e32 v9, 0x1fe, v9
	ds_load_u8 v28, v28
	ds_load_u8 v10, v10 offset:1536
	ds_load_u16 v26, v26
	ds_load_u16 v27, v27 offset:512
	ds_load_u16 v29, v9 offset:1024
	s_wait_dscnt 0x3
	v_perm_b32 v10, v10, v28, 0xc0c0004
	s_wait_dscnt 0x1
	v_lshl_or_b32 v9, v27, 16, v26
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_2)
	v_lshl_or_b32 v10, v10, 16, v29
	ds_store_b64 v24, v[9:10]
	s_branch .LBB5_10
.LBB5_22:                               ; %._crit_edge100.loopexit
	v_mov_b32_e32 v9, v11
	v_mov_b32_e32 v13, v12
.LBB5_23:                               ; %Flow207
	s_delay_alu instid0(VALU_DEP_1)
	v_and_b32_e32 v0, 8, v13
	s_mul_i32 s0, ttmp7, s2
	s_wait_alu depctr_sa_sdst(0)
	s_lshl_b32 s4, ttmp9, 4
	s_lshl_b32 s0, s0, 4
	s_wait_alu depctr_sa_sdst(0)
	s_ashr_i32 s5, s4, 31
	v_mad_co_u64_u32 v[9:10], null, s2, v0, v[9:10]
	v_mov_b32_e32 v10, 0
	s_ashr_i32 s1, s0, 31
	s_wait_alu depctr_sa_sdst(0)
	s_lshl_b64 s[4:5], s[4:5], 2
	s_lshl_b64 s[0:1], s[0:1], 2
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[0:1], s[10:11], s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[0:1], s[0:1], s[4:5]
	v_lshlrev_b64_e32 v[11:12], 2, v[9:10]
	v_add_nc_u32_e32 v9, s2, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_4)
	v_lshlrev_b64_e32 v[13:14], 2, v[9:10]
	v_add_nc_u32_e32 v9, s2, v9
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v11, vcc_lo, s0, v11
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_add_co_ci_u32_e64 v12, null, s1, v12, vcc_lo
	v_lshlrev_b64_e32 v[15:16], 2, v[9:10]
	v_add_nc_u32_e32 v9, s2, v9
	v_add_co_u32 v13, vcc_lo, s0, v13
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v14, null, s1, v14, vcc_lo
	s_delay_alu instid0(VALU_DEP_4)
	v_add_co_u32 v15, vcc_lo, s0, v15
	v_lshlrev_b64_e32 v[17:18], 2, v[9:10]
	v_add_nc_u32_e32 v9, s2, v9
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v16, null, s1, v16, vcc_lo
	s_clause 0x2
	global_store_b32 v[11:12], v1, off
	global_store_b32 v[13:14], v2, off
	global_store_b32 v[15:16], v3, off
	v_lshlrev_b64_e32 v[0:1], 2, v[9:10]
	v_add_nc_u32_e32 v9, s2, v9
	v_add_co_u32 v2, vcc_lo, s0, v17
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v3, null, s1, v18, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_4) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[11:12], 2, v[9:10]
	v_add_nc_u32_e32 v9, s2, v9
	v_add_co_u32 v0, vcc_lo, s0, v0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v1, null, s1, v1, vcc_lo
	v_lshlrev_b64_e32 v[13:14], 2, v[9:10]
	v_add_nc_u32_e32 v9, s2, v9
	v_add_co_u32 v11, vcc_lo, s0, v11
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v12, null, s1, v12, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_3) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[9:10], 2, v[9:10]
	v_add_co_u32 v13, vcc_lo, s0, v13
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v14, null, s1, v14, vcc_lo
	v_add_co_u32 v9, vcc_lo, s0, v9
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v10, null, s1, v10, vcc_lo
	s_clause 0x4
	global_store_b32 v[2:3], v4, off
	global_store_b32 v[0:1], v5, off
	global_store_b32 v[11:12], v6, off
	global_store_b32 v[13:14], v7, off
	global_store_b32 v[9:10], v8, off
	s_endpgm
.Lfunc_end5:
	.size	_Z21fused_m6_repacked_k64PK14__hip_fp8_e4m3PKhS3_Pfiii, .Lfunc_end5-_Z21fused_m6_repacked_k64PK14__hip_fp8_e4m3PKhS3_Pfiii
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel _Z21fused_m6_repacked_k64PK14__hip_fp8_e4m3PKhS3_Pfiii
		.amdhsa_group_segment_fixed_size 3584
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 44
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_dispatch_ptr 0
		.amdhsa_user_sgpr_queue_ptr 0
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 0
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_wavefront_size32 1
		.amdhsa_uses_dynamic_stack 0
		.amdhsa_enable_private_segment 0
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 1
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 42
		.amdhsa_next_free_sgpr 12
		.amdhsa_reserve_vcc 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_fp16_overflow 0
		.amdhsa_workgroup_processor_mode 1
		.amdhsa_memory_ordered 1
		.amdhsa_forward_progress 1
		.amdhsa_inst_pref_size 17
		.amdhsa_round_robin_scheduling 0
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.text
                                        ; -- End function
	.set .L_Z21fused_m6_repacked_k64PK14__hip_fp8_e4m3PKhS3_Pfiii.num_vgpr, 42
	.set .L_Z21fused_m6_repacked_k64PK14__hip_fp8_e4m3PKhS3_Pfiii.num_agpr, 0
	.set .L_Z21fused_m6_repacked_k64PK14__hip_fp8_e4m3PKhS3_Pfiii.numbered_sgpr, 12
	.set .L_Z21fused_m6_repacked_k64PK14__hip_fp8_e4m3PKhS3_Pfiii.num_named_barrier, 0
	.set .L_Z21fused_m6_repacked_k64PK14__hip_fp8_e4m3PKhS3_Pfiii.private_seg_size, 0
	.set .L_Z21fused_m6_repacked_k64PK14__hip_fp8_e4m3PKhS3_Pfiii.uses_vcc, 1
	.set .L_Z21fused_m6_repacked_k64PK14__hip_fp8_e4m3PKhS3_Pfiii.uses_flat_scratch, 0
	.set .L_Z21fused_m6_repacked_k64PK14__hip_fp8_e4m3PKhS3_Pfiii.has_dyn_sized_stack, 0
	.set .L_Z21fused_m6_repacked_k64PK14__hip_fp8_e4m3PKhS3_Pfiii.has_recursion, 0
	.set .L_Z21fused_m6_repacked_k64PK14__hip_fp8_e4m3PKhS3_Pfiii.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 2096
; TotalNumSgprs: 14
; NumVgprs: 42
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 3584 bytes/workgroup (compile time only)
; SGPRBlocks: 0
; VGPRBlocks: 5
; NumSGPRsForWavesPerEU: 14
; NumVGPRsForWavesPerEU: 42
; Occupancy: 16
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 2
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
	.text
	.protected	_Z21decode_only_scatteredPKhS0_Phiii ; -- Begin function _Z21decode_only_scatteredPKhS0_Phiii
	.globl	_Z21decode_only_scatteredPKhS0_Phiii
	.p2align	8
	.type	_Z21decode_only_scatteredPKhS0_Phiii,@function
_Z21decode_only_scatteredPKhS0_Phiii:   ; @_Z21decode_only_scatteredPKhS0_Phiii
; %bb.0:
	s_clause 0x1
	s_load_b32 s2, s[0:1], 0x34
	s_load_b96 s[4:6], s[0:1], 0x18
	s_wait_kmcnt 0x0
	s_and_b32 s2, s2, 0xffff
	s_ashr_i32 s3, s5, 31
	v_mad_co_u64_u32 v[0:1], null, ttmp9, s2, v[0:1]
	s_lshr_b32 s2, s3, 29
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s2, s5, s2
	s_wait_alu depctr_sa_sdst(0)
	s_ashr_i32 s2, s2, 3
	s_wait_alu depctr_sa_sdst(0)
	s_mul_i32 s3, s2, s4
	s_wait_alu depctr_sa_sdst(0)
	v_cmp_gt_i32_e32 vcc_lo, s3, v0
	s_and_saveexec_b32 s3, vcc_lo
	s_cbranch_execz .LBB6_2
; %bb.1:
	s_abs_i32 s3, s2
	s_load_b128 s[8:11], s[0:1], 0x0
	s_cvt_f32_u32 s4, s3
	s_sub_co_i32 s5, 0, s3
	s_load_b64 s[0:1], s[0:1], 0x10
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_iflag_f32_e32 v1, s4
	v_readfirstlane_b32 s4, v1
	v_sub_nc_u32_e32 v1, 0, v0
	s_mul_f32 s4, s4, 0x4f7ffffe
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)
	v_max_i32_e32 v1, v1, v0
	s_wait_alu depctr_sa_sdst(0)
	s_cvt_u32_f32 s4, s4
	s_wait_alu depctr_sa_sdst(0)
	s_delay_alu instid0(SALU_CYCLE_2)
	s_mul_i32 s5, s5, s4
	s_wait_alu depctr_sa_sdst(0)
	s_mul_hi_u32 s5, s4, s5
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s4, s4, s5
	s_wait_alu depctr_sa_sdst(0)
	v_mul_hi_u32 v2, v1, s4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v3, v2, s3
	v_sub_nc_u32_e32 v1, v1, v3
	v_add_nc_u32_e32 v3, 1, v2
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_subrev_nc_u32_e32 v4, s3, v1
	v_cmp_le_u32_e32 vcc_lo, s3, v1
	v_dual_cndmask_b32 v2, v2, v3 :: v_dual_cndmask_b32 v1, v1, v4
	v_xor_b32_e32 v3, s2, v0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_add_nc_u32_e32 v4, 1, v2
	v_cmp_le_u32_e32 vcc_lo, s3, v1
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_ashrrev_i32_e32 v3, 31, v3
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v1, v2, v4, vcc_lo
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_xor_b32_e32 v1, v1, v3
	v_sub_nc_u32_e32 v1, v1, v3
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_mul_lo_u32 v2, v1, s2
	v_mul_lo_u32 v1, v1, s6
	v_sub_nc_u32_e32 v2, v0, v2
	v_lshlrev_b32_e32 v0, 3, v0
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_bfe_i32 v3, v2, 28, 1
	v_lshlrev_b32_e32 v4, 3, v2
	v_ashrrev_i32_e32 v5, 31, v2
	v_lshrrev_b32_e32 v3, 24, v3
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_lshrrev_b32_e32 v5, 27, v5
	v_add_nc_u32_e32 v3, v4, v3
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_lshl_u32 v2, v2, v5, 2
	v_and_b32_e32 v3, 0xffffff00, v3
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_and_b32_e32 v2, 0xffffff80, v2
	v_sub_nc_u32_e32 v3, v4, v3
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_ashrrev_i32_e32 v3, 1, v3
	v_add3_u32 v1, v2, v1, v3
	v_ashrrev_i32_e32 v3, 31, v0
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_3) | instid1(VALU_DEP_2)
	v_ashrrev_i32_e32 v2, 31, v1
	s_wait_kmcnt 0x0
	v_add_co_u32 v1, vcc_lo, s8, v1
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v2, null, s9, v2, vcc_lo
	global_load_b32 v4, v[1:2], off
	v_add_co_u32 v2, vcc_lo, s0, v0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v3, null, s1, v3, vcc_lo
	s_wait_loadcnt 0x0
	v_lshlrev_b32_e32 v1, 1, v4
	s_delay_alu instid0(VALU_DEP_1)
	v_and_b32_e32 v5, 0x1fe, v1
	global_load_d16_u8 v1, v5, s[10:11]
	s_wait_loadcnt 0x0
	global_store_b8 v[2:3], v1, off
	global_load_d16_u8 v0, v5, s[10:11] offset:1
	v_lshrrev_b32_e32 v1, 7, v4
	s_delay_alu instid0(VALU_DEP_1)
	v_and_b32_e32 v1, 0x1fe, v1
	s_wait_loadcnt 0x0
	global_store_b8 v[2:3], v0, off offset:1
	global_load_d16_u8 v0, v1, s[10:11] offset:512
	s_wait_loadcnt 0x0
	global_store_b8 v[2:3], v0, off offset:2
	global_load_d16_u8 v0, v1, s[10:11] offset:513
	v_lshrrev_b32_e32 v1, 15, v4
	s_delay_alu instid0(VALU_DEP_1)
	v_and_b32_e32 v1, 0x1fe, v1
	s_wait_loadcnt 0x0
	global_store_b8 v[2:3], v0, off offset:3
	global_load_d16_u8 v0, v1, s[10:11] offset:1024
	s_wait_loadcnt 0x0
	global_store_b8 v[2:3], v0, off offset:4
	global_load_d16_u8 v0, v1, s[10:11] offset:1025
	v_lshrrev_b32_e32 v1, 23, v4
	s_delay_alu instid0(VALU_DEP_1)
	v_and_b32_e32 v4, 0x1fe, v1
	v_or_b32_e32 v1, 0x601, v1
	s_wait_loadcnt 0x0
	global_store_b8 v[2:3], v0, off offset:5
	global_load_d16_u8 v0, v4, s[10:11] offset:1536
	s_wait_loadcnt 0x0
	global_store_b8 v[2:3], v0, off offset:6
	global_load_d16_u8 v0, v1, s[10:11]
	s_wait_loadcnt 0x0
	global_store_b8 v[2:3], v0, off offset:7
.LBB6_2:
	s_endpgm
.Lfunc_end6:
	.size	_Z21decode_only_scatteredPKhS0_Phiii, .Lfunc_end6-_Z21decode_only_scatteredPKhS0_Phiii
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel _Z21decode_only_scatteredPKhS0_Phiii
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 296
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_dispatch_ptr 0
		.amdhsa_user_sgpr_queue_ptr 0
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 0
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_wavefront_size32 1
		.amdhsa_uses_dynamic_stack 0
		.amdhsa_enable_private_segment 0
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 0
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 6
		.amdhsa_next_free_sgpr 12
		.amdhsa_reserve_vcc 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_fp16_overflow 0
		.amdhsa_workgroup_processor_mode 1
		.amdhsa_memory_ordered 1
		.amdhsa_forward_progress 1
		.amdhsa_inst_pref_size 6
		.amdhsa_round_robin_scheduling 0
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.text
                                        ; -- End function
	.set .L_Z21decode_only_scatteredPKhS0_Phiii.num_vgpr, 6
	.set .L_Z21decode_only_scatteredPKhS0_Phiii.num_agpr, 0
	.set .L_Z21decode_only_scatteredPKhS0_Phiii.numbered_sgpr, 12
	.set .L_Z21decode_only_scatteredPKhS0_Phiii.num_named_barrier, 0
	.set .L_Z21decode_only_scatteredPKhS0_Phiii.private_seg_size, 0
	.set .L_Z21decode_only_scatteredPKhS0_Phiii.uses_vcc, 1
	.set .L_Z21decode_only_scatteredPKhS0_Phiii.uses_flat_scratch, 0
	.set .L_Z21decode_only_scatteredPKhS0_Phiii.has_dyn_sized_stack, 0
	.set .L_Z21decode_only_scatteredPKhS0_Phiii.has_recursion, 0
	.set .L_Z21decode_only_scatteredPKhS0_Phiii.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 768
; TotalNumSgprs: 14
; NumVgprs: 6
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 0
; VGPRBlocks: 0
; NumSGPRsForWavesPerEU: 14
; NumVGPRsForWavesPerEU: 6
; Occupancy: 16
; WaveLimiterHint : 1
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 2
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
	.text
	.protected	_Z20decode_only_repackedPKhS0_Phii ; -- Begin function _Z20decode_only_repackedPKhS0_Phii
	.globl	_Z20decode_only_repackedPKhS0_Phii
	.p2align	8
	.type	_Z20decode_only_repackedPKhS0_Phii,@function
_Z20decode_only_repackedPKhS0_Phii:     ; @_Z20decode_only_repackedPKhS0_Phii
; %bb.0:
	s_clause 0x1
	s_load_b32 s4, s[0:1], 0x2c
	s_load_b64 s[2:3], s[0:1], 0x18
	s_wait_kmcnt 0x0
	s_and_b32 s4, s4, 0xffff
	s_ashr_i32 s5, s3, 31
	v_mad_co_u64_u32 v[0:1], null, ttmp9, s4, v[0:1]
	s_lshr_b32 s4, s5, 29
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s3, s3, s4
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_ashr_i32 s3, s3, 3
	s_mul_i32 s3, s3, s2
	s_mov_b32 s2, exec_lo
	v_cmpx_gt_i32_e64 s3, v0
	s_cbranch_execz .LBB7_2
; %bb.1:
	v_ashrrev_i32_e32 v1, 31, v0
	s_clause 0x1
	s_load_b128 s[4:7], s[0:1], 0x0
	s_load_b64 s[0:1], s[0:1], 0x10
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshrrev_b32_e32 v1, 26, v1
	v_add_nc_u32_e32 v2, v0, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_and_b32_e32 v1, 0xffc0, v2
	v_lshlrev_b32_e32 v2, 2, v2
	v_sub_nc_u32_e32 v3, v0, v1
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_and_b32_e32 v2, 0xffffff00, v2
	v_lshlrev_b32_e32 v0, 3, v0
	v_bfe_i32 v1, v3, 0, 8
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshrrev_b16 v1.l, 13, v1.l
	v_and_b16 v1.l, v1.l, 3
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_nc_u16 v4.l, v3.l, v1.l
	v_bfe_i32 v1, v4, 0, 8
	v_and_b16 v1.h, 0xfc, v4.l
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_ashrrev_i16 v1.l, 2, v1.l
	v_sub_nc_u16 v3.l, v3.l, v1.h
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_bfe_i32 v1, v1, 0, 16
	v_bfe_i32 v3, v3, 0, 8
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshl_add_u32 v1, v1, 4, v2
	v_lshl_add_u32 v1, v3, 2, v1
	v_ashrrev_i32_e32 v3, 31, v0
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_ashrrev_i32_e32 v2, 31, v1
	s_wait_kmcnt 0x0
	v_add_co_u32 v1, vcc_lo, s4, v1
	v_add_co_ci_u32_e64 v2, null, s5, v2, vcc_lo
	global_load_b32 v4, v[1:2], off
	v_add_co_u32 v2, vcc_lo, s0, v0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v3, null, s1, v3, vcc_lo
	s_wait_loadcnt 0x0
	v_lshlrev_b32_e32 v1, 1, v4
	s_delay_alu instid0(VALU_DEP_1)
	v_and_b32_e32 v5, 0x1fe, v1
	global_load_d16_u8 v1, v5, s[6:7]
	s_wait_loadcnt 0x0
	global_store_b8 v[2:3], v1, off
	global_load_d16_u8 v0, v5, s[6:7] offset:1
	v_lshrrev_b32_e32 v1, 7, v4
	s_delay_alu instid0(VALU_DEP_1)
	v_and_b32_e32 v1, 0x1fe, v1
	s_wait_loadcnt 0x0
	global_store_b8 v[2:3], v0, off offset:1
	global_load_d16_u8 v0, v1, s[6:7] offset:512
	s_wait_loadcnt 0x0
	global_store_b8 v[2:3], v0, off offset:2
	global_load_d16_u8 v0, v1, s[6:7] offset:513
	v_lshrrev_b32_e32 v1, 15, v4
	s_delay_alu instid0(VALU_DEP_1)
	v_and_b32_e32 v1, 0x1fe, v1
	s_wait_loadcnt 0x0
	global_store_b8 v[2:3], v0, off offset:3
	global_load_d16_u8 v0, v1, s[6:7] offset:1024
	s_wait_loadcnt 0x0
	global_store_b8 v[2:3], v0, off offset:4
	global_load_d16_u8 v0, v1, s[6:7] offset:1025
	v_lshrrev_b32_e32 v1, 23, v4
	s_delay_alu instid0(VALU_DEP_1)
	v_and_b32_e32 v4, 0x1fe, v1
	v_or_b32_e32 v1, 0x601, v1
	s_wait_loadcnt 0x0
	global_store_b8 v[2:3], v0, off offset:5
	global_load_d16_u8 v0, v4, s[6:7] offset:1536
	s_wait_loadcnt 0x0
	global_store_b8 v[2:3], v0, off offset:6
	global_load_d16_u8 v0, v1, s[6:7]
	s_wait_loadcnt 0x0
	global_store_b8 v[2:3], v0, off offset:7
.LBB7_2:
	s_endpgm
.Lfunc_end7:
	.size	_Z20decode_only_repackedPKhS0_Phii, .Lfunc_end7-_Z20decode_only_repackedPKhS0_Phii
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel _Z20decode_only_repackedPKhS0_Phii
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 288
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_dispatch_ptr 0
		.amdhsa_user_sgpr_queue_ptr 0
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 0
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_wavefront_size32 1
		.amdhsa_uses_dynamic_stack 0
		.amdhsa_enable_private_segment 0
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 0
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 6
		.amdhsa_next_free_sgpr 8
		.amdhsa_reserve_vcc 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_fp16_overflow 0
		.amdhsa_workgroup_processor_mode 1
		.amdhsa_memory_ordered 1
		.amdhsa_forward_progress 1
		.amdhsa_inst_pref_size 6
		.amdhsa_round_robin_scheduling 0
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.text
                                        ; -- End function
	.set .L_Z20decode_only_repackedPKhS0_Phii.num_vgpr, 6
	.set .L_Z20decode_only_repackedPKhS0_Phii.num_agpr, 0
	.set .L_Z20decode_only_repackedPKhS0_Phii.numbered_sgpr, 8
	.set .L_Z20decode_only_repackedPKhS0_Phii.num_named_barrier, 0
	.set .L_Z20decode_only_repackedPKhS0_Phii.private_seg_size, 0
	.set .L_Z20decode_only_repackedPKhS0_Phii.uses_vcc, 1
	.set .L_Z20decode_only_repackedPKhS0_Phii.uses_flat_scratch, 0
	.set .L_Z20decode_only_repackedPKhS0_Phii.has_dyn_sized_stack, 0
	.set .L_Z20decode_only_repackedPKhS0_Phii.has_recursion, 0
	.set .L_Z20decode_only_repackedPKhS0_Phii.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 644
; TotalNumSgprs: 10
; NumVgprs: 6
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 0
; VGPRBlocks: 0
; NumSGPRsForWavesPerEU: 10
; NumVGPRsForWavesPerEU: 6
; Occupancy: 16
; WaveLimiterHint : 1
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 2
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
	.text
	.p2alignl 7, 3214868480
	.fill 96, 4, 3214868480
	.section	.AMDGPU.gpr_maximums,"",@progbits
	.set amdgpu.max_num_vgpr, 0
	.set amdgpu.max_num_agpr, 0
	.set amdgpu.max_num_sgpr, 0
	.set amdgpu.max_num_named_barrier, 0
	.text
	.type	__hip_cuid_437800c2e6994a55,@object ; @__hip_cuid_437800c2e6994a55
	.section	.bss,"aw",@nobits
	.globl	__hip_cuid_437800c2e6994a55
__hip_cuid_437800c2e6994a55:
	.byte	0                               ; 0x0
	.size	__hip_cuid_437800c2e6994a55, 1

	.ident	"AMD clang version 23.0.0git (https://github.com/ROCm/llvm-project.git 46fcb339fb61119b337f973c7ca9e710a319fdd0+PATCHED:440716f8b87be9d8e20ed910e10e5b6d14d57cf6)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym __hip_cuid_437800c2e6994a55
	.amdgpu_metadata
---
amdhsa.kernels:
  - .args:
      - .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         8
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         16
        .size:           8
        .value_kind:     global_buffer
      - .offset:         24
        .size:           4
        .value_kind:     by_value
      - .offset:         28
        .size:           4
        .value_kind:     by_value
      - .offset:         32
        .size:           4
        .value_kind:     by_value
    .gfx1250_revision: B0
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 36
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 1024
    .name:           _Z14plain_direct16PK14__hip_fp8_e4m3S1_Pfiii
    .private_segment_fixed_size: 0
    .sgpr_count:     14
    .sgpr_spill_count: 0
    .symbol:         _Z14plain_direct16PK14__hip_fp8_e4m3S1_Pfiii.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     18
    .vgpr_spill_count: 0
    .wavefront_size: 32
    .workgroup_processor_mode: 1
  - .args:
      - .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         8
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         16
        .size:           8
        .value_kind:     global_buffer
      - .offset:         24
        .size:           4
        .value_kind:     by_value
      - .offset:         28
        .size:           4
        .value_kind:     by_value
      - .offset:         32
        .size:           4
        .value_kind:     by_value
    .gfx1250_revision: B0
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 36
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 1024
    .name:           _Z14plain_direct32PK14__hip_fp8_e4m3S1_Pfiii
    .private_segment_fixed_size: 0
    .sgpr_count:     14
    .sgpr_spill_count: 0
    .symbol:         _Z14plain_direct32PK14__hip_fp8_e4m3S1_Pfiii.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     22
    .vgpr_spill_count: 0
    .wavefront_size: 32
    .workgroup_processor_mode: 1
  - .args:
      - .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         8
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         16
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         24
        .size:           8
        .value_kind:     global_buffer
      - .offset:         32
        .size:           4
        .value_kind:     by_value
      - .offset:         36
        .size:           4
        .value_kind:     by_value
      - .offset:         40
        .size:           4
        .value_kind:     by_value
      - .offset:         44
        .size:           4
        .value_kind:     by_value
    .gfx1250_revision: B0
    .group_segment_fixed_size: 3072
    .kernarg_segment_align: 8
    .kernarg_segment_size: 48
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 1024
    .name:           _Z12fused_m5_k32PK14__hip_fp8_e4m3PKhS3_Pfiiii
    .private_segment_fixed_size: 0
    .sgpr_count:     17
    .sgpr_spill_count: 0
    .symbol:         _Z12fused_m5_k32PK14__hip_fp8_e4m3PKhS3_Pfiiii.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     28
    .vgpr_spill_count: 0
    .wavefront_size: 32
    .workgroup_processor_mode: 1
  - .args:
      - .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         8
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         16
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         24
        .size:           8
        .value_kind:     global_buffer
      - .offset:         32
        .size:           4
        .value_kind:     by_value
      - .offset:         36
        .size:           4
        .value_kind:     by_value
      - .offset:         40
        .size:           4
        .value_kind:     by_value
    .gfx1250_revision: B0
    .group_segment_fixed_size: 2816
    .kernarg_segment_align: 8
    .kernarg_segment_size: 44
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 1024
    .name:           _Z21fused_m6_repacked_k32PK14__hip_fp8_e4m3PKhS3_Pfiii
    .private_segment_fixed_size: 0
    .sgpr_count:     14
    .sgpr_spill_count: 0
    .symbol:         _Z21fused_m6_repacked_k32PK14__hip_fp8_e4m3PKhS3_Pfiii.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     28
    .vgpr_spill_count: 0
    .wavefront_size: 32
    .workgroup_processor_mode: 1
  - .args:
      - .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         8
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         16
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         24
        .size:           8
        .value_kind:     global_buffer
      - .offset:         32
        .size:           4
        .value_kind:     by_value
      - .offset:         36
        .size:           4
        .value_kind:     by_value
      - .offset:         40
        .size:           4
        .value_kind:     by_value
    .gfx1250_revision: B0
    .group_segment_fixed_size: 3584
    .kernarg_segment_align: 8
    .kernarg_segment_size: 44
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 1024
    .name:           _Z24fused_m6_repacked_db_k32PK14__hip_fp8_e4m3PKhS3_Pfiii
    .private_segment_fixed_size: 0
    .sgpr_count:     15
    .sgpr_spill_count: 0
    .symbol:         _Z24fused_m6_repacked_db_k32PK14__hip_fp8_e4m3PKhS3_Pfiii.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     30
    .vgpr_spill_count: 0
    .wavefront_size: 32
    .workgroup_processor_mode: 1
  - .args:
      - .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         8
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         16
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         24
        .size:           8
        .value_kind:     global_buffer
      - .offset:         32
        .size:           4
        .value_kind:     by_value
      - .offset:         36
        .size:           4
        .value_kind:     by_value
      - .offset:         40
        .size:           4
        .value_kind:     by_value
    .gfx1250_revision: B0
    .group_segment_fixed_size: 3584
    .kernarg_segment_align: 8
    .kernarg_segment_size: 44
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 1024
    .name:           _Z21fused_m6_repacked_k64PK14__hip_fp8_e4m3PKhS3_Pfiii
    .private_segment_fixed_size: 0
    .sgpr_count:     14
    .sgpr_spill_count: 0
    .symbol:         _Z21fused_m6_repacked_k64PK14__hip_fp8_e4m3PKhS3_Pfiii.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     42
    .vgpr_spill_count: 0
    .wavefront_size: 32
    .workgroup_processor_mode: 1
  - .args:
      - .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         8
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         16
        .size:           8
        .value_kind:     global_buffer
      - .offset:         24
        .size:           4
        .value_kind:     by_value
      - .offset:         28
        .size:           4
        .value_kind:     by_value
      - .offset:         32
        .size:           4
        .value_kind:     by_value
      - .offset:         40
        .size:           4
        .value_kind:     hidden_block_count_x
      - .offset:         44
        .size:           4
        .value_kind:     hidden_block_count_y
      - .offset:         48
        .size:           4
        .value_kind:     hidden_block_count_z
      - .offset:         52
        .size:           2
        .value_kind:     hidden_group_size_x
      - .offset:         54
        .size:           2
        .value_kind:     hidden_group_size_y
      - .offset:         56
        .size:           2
        .value_kind:     hidden_group_size_z
      - .offset:         58
        .size:           2
        .value_kind:     hidden_remainder_x
      - .offset:         60
        .size:           2
        .value_kind:     hidden_remainder_y
      - .offset:         62
        .size:           2
        .value_kind:     hidden_remainder_z
      - .offset:         80
        .size:           8
        .value_kind:     hidden_global_offset_x
      - .offset:         88
        .size:           8
        .value_kind:     hidden_global_offset_y
      - .offset:         96
        .size:           8
        .value_kind:     hidden_global_offset_z
      - .offset:         104
        .size:           2
        .value_kind:     hidden_grid_dims
    .gfx1250_revision: B0
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 296
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 1024
    .name:           _Z21decode_only_scatteredPKhS0_Phiii
    .private_segment_fixed_size: 0
    .sgpr_count:     14
    .sgpr_spill_count: 0
    .symbol:         _Z21decode_only_scatteredPKhS0_Phiii.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     6
    .vgpr_spill_count: 0
    .wavefront_size: 32
    .workgroup_processor_mode: 1
  - .args:
      - .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         8
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         16
        .size:           8
        .value_kind:     global_buffer
      - .offset:         24
        .size:           4
        .value_kind:     by_value
      - .offset:         28
        .size:           4
        .value_kind:     by_value
      - .offset:         32
        .size:           4
        .value_kind:     hidden_block_count_x
      - .offset:         36
        .size:           4
        .value_kind:     hidden_block_count_y
      - .offset:         40
        .size:           4
        .value_kind:     hidden_block_count_z
      - .offset:         44
        .size:           2
        .value_kind:     hidden_group_size_x
      - .offset:         46
        .size:           2
        .value_kind:     hidden_group_size_y
      - .offset:         48
        .size:           2
        .value_kind:     hidden_group_size_z
      - .offset:         50
        .size:           2
        .value_kind:     hidden_remainder_x
      - .offset:         52
        .size:           2
        .value_kind:     hidden_remainder_y
      - .offset:         54
        .size:           2
        .value_kind:     hidden_remainder_z
      - .offset:         72
        .size:           8
        .value_kind:     hidden_global_offset_x
      - .offset:         80
        .size:           8
        .value_kind:     hidden_global_offset_y
      - .offset:         88
        .size:           8
        .value_kind:     hidden_global_offset_z
      - .offset:         96
        .size:           2
        .value_kind:     hidden_grid_dims
    .gfx1250_revision: B0
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 288
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 1024
    .name:           _Z20decode_only_repackedPKhS0_Phii
    .private_segment_fixed_size: 0
    .sgpr_count:     10
    .sgpr_spill_count: 0
    .symbol:         _Z20decode_only_repackedPKhS0_Phii.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     6
    .vgpr_spill_count: 0
    .wavefront_size: 32
    .workgroup_processor_mode: 1
amdhsa.target:   amdgcn-amd-amdhsa--gfx1201
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
