	.att_syntax
	.file	"gemm_m6_hip.cpp"
                                        # Start of file scope inline assembly
	.globl	_ZSt21ios_base_library_initv

                                        # End of file scope inline assembly
	.text
	.globl	_Z29__device_stub__plain_direct16PK14__hip_fp8_e4m3S1_Pfiii # -- Begin function _Z29__device_stub__plain_direct16PK14__hip_fp8_e4m3S1_Pfiii
	.prefalign	4, .Lfunc_end0, nop
	.type	_Z29__device_stub__plain_direct16PK14__hip_fp8_e4m3S1_Pfiii,@function
_Z29__device_stub__plain_direct16PK14__hip_fp8_e4m3S1_Pfiii: # @_Z29__device_stub__plain_direct16PK14__hip_fp8_e4m3S1_Pfiii
	.cfi_startproc
# %bb.0:
	subq	$152, %rsp
	.cfi_def_cfa_offset 160
	movq	%rdi, 88(%rsp)
	movq	%rsi, 80(%rsp)
	movq	%rdx, 72(%rsp)
	movl	%ecx, 20(%rsp)
	movl	%r8d, 16(%rsp)
	movl	%r9d, 12(%rsp)
	leaq	88(%rsp), %rax
	movq	%rax, 96(%rsp)
	leaq	80(%rsp), %rax
	movq	%rax, 104(%rsp)
	leaq	72(%rsp), %rax
	movq	%rax, 112(%rsp)
	leaq	20(%rsp), %rax
	movq	%rax, 120(%rsp)
	leaq	16(%rsp), %rax
	movq	%rax, 128(%rsp)
	leaq	12(%rsp), %rax
	movq	%rax, 136(%rsp)
	leaq	56(%rsp), %rdi
	leaq	40(%rsp), %rsi
	leaq	32(%rsp), %rdx
	leaq	24(%rsp), %rcx
	callq	__hipPopCallConfiguration@PLT
	movq	56(%rsp), %rsi
	movl	64(%rsp), %edx
	movq	40(%rsp), %rcx
	movl	48(%rsp), %r8d
	movq	_Z14plain_direct16PK14__hip_fp8_e4m3S1_Pfiii@GOTPCREL(%rip), %rdi
	leaq	96(%rsp), %r9
	pushq	24(%rsp)
	.cfi_adjust_cfa_offset 8
	pushq	40(%rsp)
	.cfi_adjust_cfa_offset 8
	callq	hipLaunchKernel@PLT
	addq	$168, %rsp
	.cfi_adjust_cfa_offset -168
	retq
.Lfunc_end0:
	.size	_Z29__device_stub__plain_direct16PK14__hip_fp8_e4m3S1_Pfiii, .Lfunc_end0-_Z29__device_stub__plain_direct16PK14__hip_fp8_e4m3S1_Pfiii
	.cfi_endproc
                                        # -- End function
	.globl	_Z29__device_stub__plain_direct32PK14__hip_fp8_e4m3S1_Pfiii # -- Begin function _Z29__device_stub__plain_direct32PK14__hip_fp8_e4m3S1_Pfiii
	.prefalign	4, .Lfunc_end1, nop
	.type	_Z29__device_stub__plain_direct32PK14__hip_fp8_e4m3S1_Pfiii,@function
_Z29__device_stub__plain_direct32PK14__hip_fp8_e4m3S1_Pfiii: # @_Z29__device_stub__plain_direct32PK14__hip_fp8_e4m3S1_Pfiii
	.cfi_startproc
# %bb.0:
	subq	$152, %rsp
	.cfi_def_cfa_offset 160
	movq	%rdi, 88(%rsp)
	movq	%rsi, 80(%rsp)
	movq	%rdx, 72(%rsp)
	movl	%ecx, 20(%rsp)
	movl	%r8d, 16(%rsp)
	movl	%r9d, 12(%rsp)
	leaq	88(%rsp), %rax
	movq	%rax, 96(%rsp)
	leaq	80(%rsp), %rax
	movq	%rax, 104(%rsp)
	leaq	72(%rsp), %rax
	movq	%rax, 112(%rsp)
	leaq	20(%rsp), %rax
	movq	%rax, 120(%rsp)
	leaq	16(%rsp), %rax
	movq	%rax, 128(%rsp)
	leaq	12(%rsp), %rax
	movq	%rax, 136(%rsp)
	leaq	56(%rsp), %rdi
	leaq	40(%rsp), %rsi
	leaq	32(%rsp), %rdx
	leaq	24(%rsp), %rcx
	callq	__hipPopCallConfiguration@PLT
	movq	56(%rsp), %rsi
	movl	64(%rsp), %edx
	movq	40(%rsp), %rcx
	movl	48(%rsp), %r8d
	movq	_Z14plain_direct32PK14__hip_fp8_e4m3S1_Pfiii@GOTPCREL(%rip), %rdi
	leaq	96(%rsp), %r9
	pushq	24(%rsp)
	.cfi_adjust_cfa_offset 8
	pushq	40(%rsp)
	.cfi_adjust_cfa_offset 8
	callq	hipLaunchKernel@PLT
	addq	$168, %rsp
	.cfi_adjust_cfa_offset -168
	retq
.Lfunc_end1:
	.size	_Z29__device_stub__plain_direct32PK14__hip_fp8_e4m3S1_Pfiii, .Lfunc_end1-_Z29__device_stub__plain_direct32PK14__hip_fp8_e4m3S1_Pfiii
	.cfi_endproc
                                        # -- End function
	.globl	_Z27__device_stub__fused_m5_k32PK14__hip_fp8_e4m3PKhS3_Pfiiii # -- Begin function _Z27__device_stub__fused_m5_k32PK14__hip_fp8_e4m3PKhS3_Pfiiii
	.prefalign	4, .Lfunc_end2, nop
	.type	_Z27__device_stub__fused_m5_k32PK14__hip_fp8_e4m3PKhS3_Pfiiii,@function
_Z27__device_stub__fused_m5_k32PK14__hip_fp8_e4m3PKhS3_Pfiiii: # @_Z27__device_stub__fused_m5_k32PK14__hip_fp8_e4m3PKhS3_Pfiiii
	.cfi_startproc
# %bb.0:
	subq	$168, %rsp
	.cfi_def_cfa_offset 176
	movq	%rdi, 88(%rsp)
	movq	%rsi, 80(%rsp)
	movq	%rdx, 72(%rsp)
	movq	%rcx, 64(%rsp)
	movl	%r8d, 12(%rsp)
	movl	%r9d, 8(%rsp)
	leaq	88(%rsp), %rax
	movq	%rax, 96(%rsp)
	leaq	80(%rsp), %rax
	movq	%rax, 104(%rsp)
	leaq	72(%rsp), %rax
	movq	%rax, 112(%rsp)
	leaq	64(%rsp), %rax
	movq	%rax, 120(%rsp)
	leaq	12(%rsp), %rax
	movq	%rax, 128(%rsp)
	leaq	8(%rsp), %rax
	movq	%rax, 136(%rsp)
	leaq	176(%rsp), %rax
	movq	%rax, 144(%rsp)
	leaq	184(%rsp), %rax
	movq	%rax, 152(%rsp)
	leaq	48(%rsp), %rdi
	leaq	32(%rsp), %rsi
	leaq	24(%rsp), %rdx
	leaq	16(%rsp), %rcx
	callq	__hipPopCallConfiguration@PLT
	movq	48(%rsp), %rsi
	movl	56(%rsp), %edx
	movq	32(%rsp), %rcx
	movl	40(%rsp), %r8d
	movq	_Z12fused_m5_k32PK14__hip_fp8_e4m3PKhS3_Pfiiii@GOTPCREL(%rip), %rdi
	leaq	96(%rsp), %r9
	pushq	16(%rsp)
	.cfi_adjust_cfa_offset 8
	pushq	32(%rsp)
	.cfi_adjust_cfa_offset 8
	callq	hipLaunchKernel@PLT
	addq	$184, %rsp
	.cfi_adjust_cfa_offset -184
	retq
.Lfunc_end2:
	.size	_Z27__device_stub__fused_m5_k32PK14__hip_fp8_e4m3PKhS3_Pfiiii, .Lfunc_end2-_Z27__device_stub__fused_m5_k32PK14__hip_fp8_e4m3PKhS3_Pfiiii
	.cfi_endproc
                                        # -- End function
	.globl	_Z36__device_stub__fused_m6_repacked_k32PK14__hip_fp8_e4m3PKhS3_Pfiii # -- Begin function _Z36__device_stub__fused_m6_repacked_k32PK14__hip_fp8_e4m3PKhS3_Pfiii
	.prefalign	4, .Lfunc_end3, nop
	.type	_Z36__device_stub__fused_m6_repacked_k32PK14__hip_fp8_e4m3PKhS3_Pfiii,@function
_Z36__device_stub__fused_m6_repacked_k32PK14__hip_fp8_e4m3PKhS3_Pfiii: # @_Z36__device_stub__fused_m6_repacked_k32PK14__hip_fp8_e4m3PKhS3_Pfiii
	.cfi_startproc
# %bb.0:
	subq	$152, %rsp
	.cfi_def_cfa_offset 160
	movq	%rdi, 88(%rsp)
	movq	%rsi, 80(%rsp)
	movq	%rdx, 72(%rsp)
	movq	%rcx, 64(%rsp)
	movl	%r8d, 12(%rsp)
	movl	%r9d, 8(%rsp)
	leaq	88(%rsp), %rax
	movq	%rax, 96(%rsp)
	leaq	80(%rsp), %rax
	movq	%rax, 104(%rsp)
	leaq	72(%rsp), %rax
	movq	%rax, 112(%rsp)
	leaq	64(%rsp), %rax
	movq	%rax, 120(%rsp)
	leaq	12(%rsp), %rax
	movq	%rax, 128(%rsp)
	leaq	8(%rsp), %rax
	movq	%rax, 136(%rsp)
	leaq	160(%rsp), %rax
	movq	%rax, 144(%rsp)
	leaq	48(%rsp), %rdi
	leaq	32(%rsp), %rsi
	leaq	24(%rsp), %rdx
	leaq	16(%rsp), %rcx
	callq	__hipPopCallConfiguration@PLT
	movq	48(%rsp), %rsi
	movl	56(%rsp), %edx
	movq	32(%rsp), %rcx
	movl	40(%rsp), %r8d
	movq	_Z21fused_m6_repacked_k32PK14__hip_fp8_e4m3PKhS3_Pfiii@GOTPCREL(%rip), %rdi
	leaq	96(%rsp), %r9
	pushq	16(%rsp)
	.cfi_adjust_cfa_offset 8
	pushq	32(%rsp)
	.cfi_adjust_cfa_offset 8
	callq	hipLaunchKernel@PLT
	addq	$168, %rsp
	.cfi_adjust_cfa_offset -168
	retq
.Lfunc_end3:
	.size	_Z36__device_stub__fused_m6_repacked_k32PK14__hip_fp8_e4m3PKhS3_Pfiii, .Lfunc_end3-_Z36__device_stub__fused_m6_repacked_k32PK14__hip_fp8_e4m3PKhS3_Pfiii
	.cfi_endproc
                                        # -- End function
	.globl	_Z39__device_stub__fused_m6_repacked_db_k32PK14__hip_fp8_e4m3PKhS3_Pfiii # -- Begin function _Z39__device_stub__fused_m6_repacked_db_k32PK14__hip_fp8_e4m3PKhS3_Pfiii
	.prefalign	4, .Lfunc_end4, nop
	.type	_Z39__device_stub__fused_m6_repacked_db_k32PK14__hip_fp8_e4m3PKhS3_Pfiii,@function
_Z39__device_stub__fused_m6_repacked_db_k32PK14__hip_fp8_e4m3PKhS3_Pfiii: # @_Z39__device_stub__fused_m6_repacked_db_k32PK14__hip_fp8_e4m3PKhS3_Pfiii
	.cfi_startproc
# %bb.0:
	subq	$152, %rsp
	.cfi_def_cfa_offset 160
	movq	%rdi, 88(%rsp)
	movq	%rsi, 80(%rsp)
	movq	%rdx, 72(%rsp)
	movq	%rcx, 64(%rsp)
	movl	%r8d, 12(%rsp)
	movl	%r9d, 8(%rsp)
	leaq	88(%rsp), %rax
	movq	%rax, 96(%rsp)
	leaq	80(%rsp), %rax
	movq	%rax, 104(%rsp)
	leaq	72(%rsp), %rax
	movq	%rax, 112(%rsp)
	leaq	64(%rsp), %rax
	movq	%rax, 120(%rsp)
	leaq	12(%rsp), %rax
	movq	%rax, 128(%rsp)
	leaq	8(%rsp), %rax
	movq	%rax, 136(%rsp)
	leaq	160(%rsp), %rax
	movq	%rax, 144(%rsp)
	leaq	48(%rsp), %rdi
	leaq	32(%rsp), %rsi
	leaq	24(%rsp), %rdx
	leaq	16(%rsp), %rcx
	callq	__hipPopCallConfiguration@PLT
	movq	48(%rsp), %rsi
	movl	56(%rsp), %edx
	movq	32(%rsp), %rcx
	movl	40(%rsp), %r8d
	movq	_Z24fused_m6_repacked_db_k32PK14__hip_fp8_e4m3PKhS3_Pfiii@GOTPCREL(%rip), %rdi
	leaq	96(%rsp), %r9
	pushq	16(%rsp)
	.cfi_adjust_cfa_offset 8
	pushq	32(%rsp)
	.cfi_adjust_cfa_offset 8
	callq	hipLaunchKernel@PLT
	addq	$168, %rsp
	.cfi_adjust_cfa_offset -168
	retq
.Lfunc_end4:
	.size	_Z39__device_stub__fused_m6_repacked_db_k32PK14__hip_fp8_e4m3PKhS3_Pfiii, .Lfunc_end4-_Z39__device_stub__fused_m6_repacked_db_k32PK14__hip_fp8_e4m3PKhS3_Pfiii
	.cfi_endproc
                                        # -- End function
	.globl	_Z36__device_stub__fused_m6_repacked_k64PK14__hip_fp8_e4m3PKhS3_Pfiii # -- Begin function _Z36__device_stub__fused_m6_repacked_k64PK14__hip_fp8_e4m3PKhS3_Pfiii
	.prefalign	4, .Lfunc_end5, nop
	.type	_Z36__device_stub__fused_m6_repacked_k64PK14__hip_fp8_e4m3PKhS3_Pfiii,@function
_Z36__device_stub__fused_m6_repacked_k64PK14__hip_fp8_e4m3PKhS3_Pfiii: # @_Z36__device_stub__fused_m6_repacked_k64PK14__hip_fp8_e4m3PKhS3_Pfiii
	.cfi_startproc
# %bb.0:
	subq	$152, %rsp
	.cfi_def_cfa_offset 160
	movq	%rdi, 88(%rsp)
	movq	%rsi, 80(%rsp)
	movq	%rdx, 72(%rsp)
	movq	%rcx, 64(%rsp)
	movl	%r8d, 12(%rsp)
	movl	%r9d, 8(%rsp)
	leaq	88(%rsp), %rax
	movq	%rax, 96(%rsp)
	leaq	80(%rsp), %rax
	movq	%rax, 104(%rsp)
	leaq	72(%rsp), %rax
	movq	%rax, 112(%rsp)
	leaq	64(%rsp), %rax
	movq	%rax, 120(%rsp)
	leaq	12(%rsp), %rax
	movq	%rax, 128(%rsp)
	leaq	8(%rsp), %rax
	movq	%rax, 136(%rsp)
	leaq	160(%rsp), %rax
	movq	%rax, 144(%rsp)
	leaq	48(%rsp), %rdi
	leaq	32(%rsp), %rsi
	leaq	24(%rsp), %rdx
	leaq	16(%rsp), %rcx
	callq	__hipPopCallConfiguration@PLT
	movq	48(%rsp), %rsi
	movl	56(%rsp), %edx
	movq	32(%rsp), %rcx
	movl	40(%rsp), %r8d
	movq	_Z21fused_m6_repacked_k64PK14__hip_fp8_e4m3PKhS3_Pfiii@GOTPCREL(%rip), %rdi
	leaq	96(%rsp), %r9
	pushq	16(%rsp)
	.cfi_adjust_cfa_offset 8
	pushq	32(%rsp)
	.cfi_adjust_cfa_offset 8
	callq	hipLaunchKernel@PLT
	addq	$168, %rsp
	.cfi_adjust_cfa_offset -168
	retq
.Lfunc_end5:
	.size	_Z36__device_stub__fused_m6_repacked_k64PK14__hip_fp8_e4m3PKhS3_Pfiii, .Lfunc_end5-_Z36__device_stub__fused_m6_repacked_k64PK14__hip_fp8_e4m3PKhS3_Pfiii
	.cfi_endproc
                                        # -- End function
	.globl	_Z17repack_k32_for_m6PKhPhiii   # -- Begin function _Z17repack_k32_for_m6PKhPhiii
	.prefalign	4, .Lfunc_end6, nop
	.type	_Z17repack_k32_for_m6PKhPhiii,@function
_Z17repack_k32_for_m6PKhPhiii:          # @_Z17repack_k32_for_m6PKhPhiii
	.cfi_startproc
# %bb.0:
                                        # kill: def $ecx killed $ecx def $rcx
                                        # kill: def $edx killed $edx def $rdx
	cmpl	$16, %edx
	jl	.LBB6_9
# %bb.1:                                # %.preheader.lr.ph
	pushq	%rbp
	.cfi_def_cfa_offset 16
	pushq	%r15
	.cfi_def_cfa_offset 24
	pushq	%r14
	.cfi_def_cfa_offset 32
	pushq	%r13
	.cfi_def_cfa_offset 40
	pushq	%r12
	.cfi_def_cfa_offset 48
	pushq	%rbx
	.cfi_def_cfa_offset 56
	.cfi_offset %rbx, -56
	.cfi_offset %r12, -48
	.cfi_offset %r13, -40
	.cfi_offset %r14, -32
	.cfi_offset %r15, -24
	.cfi_offset %rbp, -16
	leal	31(%rcx), %eax
	testl	%ecx, %ecx
	cmovnsl	%ecx, %eax
	sarl	$5, %eax
	shrl	$4, %edx
	movslq	%r8d, %r8
	addq	$12, %rsi
	movq	%rdx, %r9
	shlq	$8, %r9
	movq	%r8, %r10
	shlq	$4, %r10
	xorl	%r11d, %r11d
	jmp	.LBB6_2
	.p2align	4
.LBB6_7:                                # %._crit_edge
                                        #   in Loop: Header=BB6_2 Depth=1
	incq	%r11
	addq	$256, %rsi                      # imm = 0x100
	addq	%r10, %rdi
	cmpq	%rdx, %r11
	je	.LBB6_8
.LBB6_2:                                # %.preheader
                                        # =>This Loop Header: Depth=1
                                        #     Child Loop BB6_4 Depth 2
                                        #       Child Loop BB6_5 Depth 3
	cmpl	$32, %ecx
	jl	.LBB6_7
# %bb.3:                                # %.lr.ph
                                        #   in Loop: Header=BB6_2 Depth=1
	xorl	%ebx, %ebx
	movq	%rsi, %r14
	xorl	%r15d, %r15d
	.p2align	4
.LBB6_4:                                #   Parent Loop BB6_2 Depth=1
                                        # =>  This Loop Header: Depth=2
                                        #       Child Loop BB6_5 Depth 3
	movl	%ebx, %ebp
	andl	$1073741696, %ebp               # imm = 0x3FFFFF80
	movl	%r15d, %r12d
	andl	$7, %r12d
	shll	$4, %r12d
	orl	%ebp, %r12d
	addq	%rdi, %r12
	xorl	%r13d, %r13d
	.p2align	4
.LBB6_5:                                #   Parent Loop BB6_2 Depth=1
                                        #     Parent Loop BB6_4 Depth=2
                                        # =>    This Inner Loop Header: Depth=3
	movl	(%r12), %ebp
	movl	%ebp, -12(%r14,%r13)
	movl	4(%r12), %ebp
	movl	%ebp, -8(%r14,%r13)
	movl	8(%r12), %ebp
	movl	%ebp, -4(%r14,%r13)
	movl	12(%r12), %ebp
	movl	%ebp, (%r14,%r13)
	addq	$16, %r13
	addq	%r8, %r12
	cmpq	$256, %r13                      # imm = 0x100
	jne	.LBB6_5
# %bb.6:                                #   in Loop: Header=BB6_4 Depth=2
	incq	%r15
	addq	%r9, %r14
	addq	$16, %rbx
	cmpq	%rax, %r15
	jne	.LBB6_4
	jmp	.LBB6_7
.LBB6_8:
	popq	%rbx
	.cfi_def_cfa_offset 48
	popq	%r12
	.cfi_def_cfa_offset 40
	popq	%r13
	.cfi_def_cfa_offset 32
	popq	%r14
	.cfi_def_cfa_offset 24
	popq	%r15
	.cfi_def_cfa_offset 16
	popq	%rbp
	.cfi_def_cfa_offset 8
	.cfi_restore %rbx
	.cfi_restore %r12
	.cfi_restore %r13
	.cfi_restore %r14
	.cfi_restore %r15
	.cfi_restore %rbp
.LBB6_9:                                # %._crit_edge39
	retq
.Lfunc_end6:
	.size	_Z17repack_k32_for_m6PKhPhiii, .Lfunc_end6-_Z17repack_k32_for_m6PKhPhiii
	.cfi_endproc
                                        # -- End function
	.globl	_Z17repack_k64_for_m6PKhPhiii   # -- Begin function _Z17repack_k64_for_m6PKhPhiii
	.prefalign	4, .Lfunc_end7, nop
	.type	_Z17repack_k64_for_m6PKhPhiii,@function
_Z17repack_k64_for_m6PKhPhiii:          # @_Z17repack_k64_for_m6PKhPhiii
	.cfi_startproc
# %bb.0:
                                        # kill: def $ecx killed $ecx def $rcx
                                        # kill: def $edx killed $edx def $rdx
	cmpl	$16, %edx
	jl	.LBB7_9
# %bb.1:                                # %.preheader.lr.ph
	pushq	%rbp
	.cfi_def_cfa_offset 16
	pushq	%r15
	.cfi_def_cfa_offset 24
	pushq	%r14
	.cfi_def_cfa_offset 32
	pushq	%r13
	.cfi_def_cfa_offset 40
	pushq	%r12
	.cfi_def_cfa_offset 48
	pushq	%rbx
	.cfi_def_cfa_offset 56
	.cfi_offset %rbx, -56
	.cfi_offset %r12, -48
	.cfi_offset %r13, -40
	.cfi_offset %r14, -32
	.cfi_offset %r15, -24
	.cfi_offset %rbp, -16
	leal	63(%rcx), %eax
	testl	%ecx, %ecx
	cmovnsl	%ecx, %eax
	sarl	$6, %eax
	shrl	$4, %edx
	movslq	%r8d, %r8
	addq	$28, %rsi
	movq	%rdx, %r9
	shlq	$9, %r9
	addq	$16, %rdi
	movq	%r8, %r10
	shlq	$4, %r10
	xorl	%r11d, %r11d
	jmp	.LBB7_2
	.p2align	4
.LBB7_7:                                # %._crit_edge
                                        #   in Loop: Header=BB7_2 Depth=1
	incq	%r11
	addq	$512, %rsi                      # imm = 0x200
	addq	%r10, %rdi
	cmpq	%rdx, %r11
	je	.LBB7_8
.LBB7_2:                                # %.preheader
                                        # =>This Loop Header: Depth=1
                                        #     Child Loop BB7_4 Depth 2
                                        #       Child Loop BB7_5 Depth 3
	cmpl	$64, %ecx
	jl	.LBB7_7
# %bb.3:                                # %.lr.ph
                                        #   in Loop: Header=BB7_2 Depth=1
	xorl	%ebx, %ebx
	movq	%rsi, %r14
	xorl	%r15d, %r15d
	.p2align	4
.LBB7_4:                                #   Parent Loop BB7_2 Depth=1
                                        # =>  This Loop Header: Depth=2
                                        #       Child Loop BB7_5 Depth 3
	movl	%ebx, %r12d
	shrl	%r12d
	andl	$1073741792, %r12d              # imm = 0x3FFFFFE0
	addq	%rdi, %r12
	xorl	%r13d, %r13d
	.p2align	4
.LBB7_5:                                #   Parent Loop BB7_2 Depth=1
                                        #     Parent Loop BB7_4 Depth=2
                                        # =>    This Inner Loop Header: Depth=3
	movl	-16(%r12), %ebp
	movl	%ebp, -28(%r14,%r13)
	movl	-12(%r12), %ebp
	movl	%ebp, -24(%r14,%r13)
	movl	-8(%r12), %ebp
	movl	%ebp, -20(%r14,%r13)
	movl	-4(%r12), %ebp
	movl	%ebp, -16(%r14,%r13)
	movl	(%r12), %ebp
	movl	%ebp, -12(%r14,%r13)
	movl	4(%r12), %ebp
	movl	%ebp, -8(%r14,%r13)
	movl	8(%r12), %ebp
	movl	%ebp, -4(%r14,%r13)
	movl	12(%r12), %ebp
	movl	%ebp, (%r14,%r13)
	addq	$32, %r13
	addq	%r8, %r12
	cmpq	$512, %r13                      # imm = 0x200
	jne	.LBB7_5
# %bb.6:                                #   in Loop: Header=BB7_4 Depth=2
	incq	%r15
	addq	%r9, %r14
	addq	$64, %rbx
	cmpq	%rax, %r15
	jne	.LBB7_4
	jmp	.LBB7_7
.LBB7_8:
	popq	%rbx
	.cfi_def_cfa_offset 48
	popq	%r12
	.cfi_def_cfa_offset 40
	popq	%r13
	.cfi_def_cfa_offset 32
	popq	%r14
	.cfi_def_cfa_offset 24
	popq	%r15
	.cfi_def_cfa_offset 16
	popq	%rbp
	.cfi_def_cfa_offset 8
	.cfi_restore %rbx
	.cfi_restore %r12
	.cfi_restore %r13
	.cfi_restore %r14
	.cfi_restore %r15
	.cfi_restore %rbp
.LBB7_9:                                # %._crit_edge39
	retq
.Lfunc_end7:
	.size	_Z17repack_k64_for_m6PKhPhiii, .Lfunc_end7-_Z17repack_k64_for_m6PKhPhiii
	.cfi_endproc
                                        # -- End function
	.section	.rodata.cst4,"aM",@progbits,4
	.p2align	2, 0x0                          # -- Begin function _Z11bench_plainiiiii
.LCPI8_0:
	.long	0x3f000000                      # float 0.5
	.text
	.globl	_Z11bench_plainiiiii
	.prefalign	4, .Lfunc_end8, nop
	.type	_Z11bench_plainiiiii,@function
_Z11bench_plainiiiii:                   # @_Z11bench_plainiiiii
.Lfunc_begin0:
	.cfi_startproc
	.cfi_personality 155, DW.ref.__gxx_personality_v0
	.cfi_lsda 27, .Lexception0
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	pushq	%r15
	.cfi_def_cfa_offset 24
	pushq	%r14
	.cfi_def_cfa_offset 32
	pushq	%r13
	.cfi_def_cfa_offset 40
	pushq	%r12
	.cfi_def_cfa_offset 48
	pushq	%rbx
	.cfi_def_cfa_offset 56
	subq	$5272, %rsp                     # imm = 0x1498
	.cfi_def_cfa_offset 5328
	.cfi_offset %rbx, -56
	.cfi_offset %r12, -48
	.cfi_offset %r13, -40
	.cfi_offset %r14, -32
	.cfi_offset %r15, -24
	.cfi_offset %rbp, -16
	movl	%r8d, 124(%rsp)                 # 4-byte Spill
	movl	%ecx, 20(%rsp)                  # 4-byte Spill
	movl	%edx, 16(%rsp)                  # 4-byte Spill
                                        # kill: def $esi killed $esi def $rsi
	movq	%rsi, 40(%rsp)                  # 8-byte Spill
                                        # kill: def $edi killed $edi def $rdi
	movq	%rdi, 48(%rsp)                  # 8-byte Spill
	movq	$1, 272(%rsp)
	movl	$1, %ecx
	movl	$2, %eax
	.p2align	4
.LBB8_1:                                # =>This Inner Loop Header: Depth=1
	movq	%rcx, %rdx
	shrq	$30, %rdx
	xorq	%rcx, %rdx
	imulq	$1812433253, %rdx, %rcx         # imm = 0x6C078965
	addq	%rax, %rcx
	decq	%rcx
	movl	%ecx, %edx
	movq	%rdx, 264(%rsp,%rax,8)
	cmpq	$624, %rax                      # imm = 0x270
	je	.LBB8_3
# %bb.2:                                #   in Loop: Header=BB8_1 Depth=1
	shrl	$30, %edx
	xorl	%edx, %ecx
	imull	$1812433253, %ecx, %ecx         # imm = 0x6C078965
	addl	%eax, %ecx
	movq	%rcx, 272(%rsp,%rax,8)
	addq	$2, %rax
	jmp	.LBB8_1
.LBB8_3:                                # %_ZNSt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EEC2Em.exit
	movq	$624, 5264(%rsp)                # imm = 0x270
	movl	16(%rsp), %eax                  # 4-byte Reload
	imull	48(%rsp), %eax                  # 4-byte Folded Reload
	testl	%eax, %eax
	js	.LBB8_93
# %bb.4:                                # %_ZNSt6vectorI14__hip_fp8_e4m3SaIS0_EE17_S_check_init_lenEmRKS1_.exit.i
	movslq	%eax, %rbx
	movq	%rbx, 264(%rsp)                 # 8-byte Spill
	je	.LBB8_5
# %bb.6:                                # %.noexc96
	.cfi_escape 0x2e, 0x00
	movq	%rbx, %rdi
	callq	_Znwm@PLT
	movq	%rbx, %rdx
	movq	%rax, %rbp
	addq	%rax, %rbx
	movb	$0, (%rax)
	movq	%rax, %r15
	incq	%r15
	decq	%rdx
	movq	%rbx, 248(%rsp)                 # 8-byte Spill
	je	.LBB8_8
# %bb.7:                                # %.lr.ph.preheader.i.i.i.i.i.i.i.i.i
	.cfi_escape 0x2e, 0x00
	movq	%r15, %rdi
	xorl	%esi, %esi
	callq	memset@PLT
	movq	%rbx, %r15
	jmp	.LBB8_8
.LBB8_5:
	movq	$0, 248(%rsp)                   # 8-byte Folded Spill
	xorl	%ebp, %ebp
	xorl	%r15d, %r15d
.LBB8_8:                                # %_ZNSt6vectorI14__hip_fp8_e4m3SaIS0_EEC2EmRKS1_.exit
	movl	16(%rsp), %eax                  # 4-byte Reload
	imull	40(%rsp), %eax                  # 4-byte Folded Reload
	testl	%eax, %eax
	js	.LBB8_9
# %bb.11:                               # %_ZNSt6vectorI14__hip_fp8_e4m3SaIS0_EE17_S_check_init_lenEmRKS1_.exit.i97
	movslq	%eax, %r12
	movq	%r12, 256(%rsp)                 # 8-byte Spill
	je	.LBB8_12
# %bb.13:
.Ltmp0:                                 # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movq	%r12, %rdi
	callq	_Znwm@PLT
.Ltmp1:                                 # EH_LABEL
# %bb.14:                               # %.noexc103
	leaq	(%rax,%r12), %r14
	movb	$0, (%rax)
	movq	%rax, 136(%rsp)                 # 8-byte Spill
	movq	%rax, %rbx
	incq	%rbx
	decq	%r12
	movq	%r14, 240(%rsp)                 # 8-byte Spill
	je	.LBB8_16
# %bb.15:                               # %.lr.ph.preheader.i.i.i.i.i.i.i.i.i99
	.cfi_escape 0x2e, 0x00
	movq	%rbx, %rdi
	xorl	%esi, %esi
	movq	%r12, %rdx
	callq	memset@PLT
	movq	%r14, %rbx
	jmp	.LBB8_16
.LBB8_12:
	movq	$0, 240(%rsp)                   # 8-byte Folded Spill
	movq	$0, 136(%rsp)                   # 8-byte Folded Spill
	xorl	%ebx, %ebx
.LBB8_16:                               # %_ZNSt6vectorI14__hip_fp8_e4m3SaIS0_EEC2EmRKS1_.exit104
	movq	%rbp, 32(%rsp)                  # 8-byte Spill
	cmpq	%r15, %rbp
	je	.LBB8_21
# %bb.17:
	leaq	272(%rsp), %r12
	movabsq	$-2049638230412172401, %r14     # imm = 0xE38E38E38E38E38F
	leaq	176(%rsp), %rbp
	movq	32(%rsp), %r13                  # 8-byte Reload
	.p2align	4
.LBB8_18:                               # %.lr.ph
                                        # =>This Inner Loop Header: Depth=1
.Ltmp2:                                 # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movq	%r12, %rdi
	callq	_ZNSt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EEclEv
.Ltmp3:                                 # EH_LABEL
# %bb.19:                               #   in Loop: Header=BB8_18 Depth=1
	movq	%rax, %rcx
	mulq	%r14
	shrq	$3, %rdx
	leaq	(%rdx,%rdx,8), %rax
	negq	%rax
	addq	%rcx, %rax
	addq	$-4, %rax
	testq	%rax, %rax
	js	.LBB8_20
# %bb.27:                               #   in Loop: Header=BB8_18 Depth=1
	xorps	%xmm0, %xmm0
	cvtsi2ss	%rax, %xmm0
	jmp	.LBB8_28
	.p2align	4
.LBB8_20:                               #   in Loop: Header=BB8_18 Depth=1
	movq	%rax, %rcx
	shrq	%rcx
	andl	$1, %eax
	orq	%rcx, %rax
	xorps	%xmm0, %xmm0
	cvtsi2ss	%rax, %xmm0
	addss	%xmm0, %xmm0
.LBB8_28:                               #   in Loop: Header=BB8_18 Depth=1
	mulss	.LCPI8_0(%rip), %xmm0
.Ltmp5:                                 # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movq	%rbp, %rdi
	callq	_ZN14__hip_fp8_e4m3C2Ef
.Ltmp6:                                 # EH_LABEL
# %bb.29:                               #   in Loop: Header=BB8_18 Depth=1
	movzbl	176(%rsp), %eax
	movb	%al, (%r13)
	incq	%r13
	cmpq	%r15, %r13
	jne	.LBB8_18
.LBB8_21:                               # %.preheader183
	cmpq	%rbx, 136(%rsp)                 # 8-byte Folded Reload
	movq	32(%rsp), %rbp                  # 8-byte Reload
	je	.LBB8_35
# %bb.22:
	leaq	272(%rsp), %r15
	movabsq	$-2049638230412172401, %r14     # imm = 0xE38E38E38E38E38F
	leaq	176(%rsp), %r12
	movq	136(%rsp), %r13                 # 8-byte Reload
	.p2align	4
.LBB8_23:                               # %.lr.ph188
                                        # =>This Inner Loop Header: Depth=1
.Ltmp8:                                 # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movq	%r15, %rdi
	callq	_ZNSt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EEclEv
.Ltmp9:                                 # EH_LABEL
# %bb.24:                               #   in Loop: Header=BB8_23 Depth=1
	movq	%rax, %rcx
	mulq	%r14
	shrq	$3, %rdx
	leaq	(%rdx,%rdx,8), %rax
	negq	%rax
	addq	%rcx, %rax
	addq	$-4, %rax
	testq	%rax, %rax
	js	.LBB8_25
# %bb.32:                               #   in Loop: Header=BB8_23 Depth=1
	xorps	%xmm0, %xmm0
	cvtsi2ss	%rax, %xmm0
	jmp	.LBB8_33
	.p2align	4
.LBB8_25:                               #   in Loop: Header=BB8_23 Depth=1
	movq	%rax, %rcx
	shrq	%rcx
	andl	$1, %eax
	orq	%rcx, %rax
	xorps	%xmm0, %xmm0
	cvtsi2ss	%rax, %xmm0
	addss	%xmm0, %xmm0
.LBB8_33:                               #   in Loop: Header=BB8_23 Depth=1
	mulss	.LCPI8_0(%rip), %xmm0
.Ltmp11:                                # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movq	%r12, %rdi
	callq	_ZN14__hip_fp8_e4m3C2Ef
.Ltmp12:                                # EH_LABEL
# %bb.34:                               #   in Loop: Header=BB8_23 Depth=1
	movzbl	176(%rsp), %eax
	movb	%al, (%r13)
	incq	%r13
	cmpq	%rbx, %r13
	jne	.LBB8_23
.LBB8_35:                               # %._crit_edge
.Ltmp14:                                # EH_LABEL
	.cfi_escape 0x2e, 0x00
	leaq	160(%rsp), %rdi
	movq	264(%rsp), %rsi                 # 8-byte Reload
	callq	hipMalloc@PLT
.Ltmp15:                                # EH_LABEL
# %bb.36:                               # %_ZL9hipMallocI14__hip_fp8_e4m3E10hipError_tPPT_m.exit
.Ltmp16:                                # EH_LABEL
	.cfi_escape 0x2e, 0x00
	leaq	152(%rsp), %rdi
	movq	256(%rsp), %rsi                 # 8-byte Reload
	callq	hipMalloc@PLT
.Ltmp17:                                # EH_LABEL
# %bb.37:                               # %_ZL9hipMallocI14__hip_fp8_e4m3E10hipError_tPPT_m.exit107
	movq	40(%rsp), %rax                  # 8-byte Reload
                                        # kill: def $eax killed $eax killed $rax
	imull	48(%rsp), %eax                  # 4-byte Folded Reload
	movslq	%eax, %r12
	shlq	$2, %r12
.Ltmp18:                                # EH_LABEL
	.cfi_escape 0x2e, 0x00
	leaq	144(%rsp), %rdi
	movq	%r12, %rsi
	callq	hipMalloc@PLT
.Ltmp19:                                # EH_LABEL
# %bb.38:                               # %_ZL9hipMallocIfE10hipError_tPPT_m.exit
	movq	160(%rsp), %rdi
.Ltmp20:                                # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movq	%rbp, %rsi
	movq	264(%rsp), %rdx                 # 8-byte Reload
	movl	$1, %ecx
	callq	hipMemcpy@PLT
.Ltmp21:                                # EH_LABEL
# %bb.39:
	movq	152(%rsp), %rdi
.Ltmp22:                                # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movq	136(%rsp), %rsi                 # 8-byte Reload
	movq	256(%rsp), %rdx                 # 8-byte Reload
	movl	$1, %ecx
	callq	hipMemcpy@PLT
.Ltmp23:                                # EH_LABEL
# %bb.40:
	movq	40(%rsp), %rcx                  # 8-byte Reload
	leal	15(%rcx), %eax
	addl	$30, %ecx
	testl	%eax, %eax
	cmovnsl	%eax, %ecx
	sarl	$4, %ecx
	movq	48(%rsp), %rdx                  # 8-byte Reload
	leal	15(%rdx), %eax
	leal	30(%rdx), %r14d
	testl	%eax, %eax
	cmovnsl	%eax, %r14d
	sarl	$4, %r14d
	shlq	$32, %r14
	orq	%rcx, %r14
	movl	$20, %ebp
	movabsq	$4294967328, %rbx               # imm = 0x100000020
	leaq	232(%rsp), %r13
	leaq	176(%rsp), %r15
	.p2align	4
.LBB8_41:                               # =>This Inner Loop Header: Depth=1
	movq	144(%rsp), %rdi
.Ltmp25:                                # EH_LABEL
	.cfi_escape 0x2e, 0x00
	xorl	%esi, %esi
	movq	%r12, %rdx
	callq	hipMemset@PLT
.Ltmp26:                                # EH_LABEL
# %bb.42:                               #   in Loop: Header=BB8_41 Depth=1
	cmpl	$0, 20(%rsp)                    # 4-byte Folded Reload
	je	.LBB8_51
# %bb.43:                               #   in Loop: Header=BB8_41 Depth=1
.Ltmp27:                                # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movq	%r14, %rdi
	movl	$1, %esi
	movq	%rbx, %rdx
	movl	$1, %ecx
	xorl	%r8d, %r8d
	xorl	%r9d, %r9d
	callq	__hipPushCallConfiguration@PLT
.Ltmp28:                                # EH_LABEL
# %bb.44:                               #   in Loop: Header=BB8_41 Depth=1
	testl	%eax, %eax
	jne	.LBB8_55
# %bb.45:                               #   in Loop: Header=BB8_41 Depth=1
	movq	160(%rsp), %rax
	movq	152(%rsp), %rcx
	movq	144(%rsp), %rdx
	movq	%rax, 232(%rsp)
	movq	%rcx, 112(%rsp)
	movq	%rdx, 104(%rsp)
	movq	48(%rsp), %rax                  # 8-byte Reload
	movl	%eax, 56(%rsp)
	movq	40(%rsp), %rax                  # 8-byte Reload
	movl	%eax, 24(%rsp)
	movl	16(%rsp), %eax                  # 4-byte Reload
	movl	%eax, 12(%rsp)
	movq	%r13, 176(%rsp)
	leaq	112(%rsp), %rax
	movq	%rax, 184(%rsp)
	leaq	104(%rsp), %rax
	movq	%rax, 192(%rsp)
	leaq	56(%rsp), %rax
	movq	%rax, 200(%rsp)
	leaq	24(%rsp), %rax
	movq	%rax, 208(%rsp)
	leaq	12(%rsp), %rax
	movq	%rax, 216(%rsp)
.Ltmp29:                                # EH_LABEL
	.cfi_escape 0x2e, 0x00
	leaq	88(%rsp), %rdi
	leaq	72(%rsp), %rsi
	leaq	64(%rsp), %rdx
	leaq	168(%rsp), %rcx
	callq	__hipPopCallConfiguration@PLT
.Ltmp30:                                # EH_LABEL
# %bb.46:                               # %.noexc109
                                        #   in Loop: Header=BB8_41 Depth=1
	movq	88(%rsp), %rsi
	movl	96(%rsp), %edx
	movq	72(%rsp), %rcx
	movl	80(%rsp), %r8d
.Ltmp31:                                # EH_LABEL
	.cfi_escape 0x2e, 0x10
	movq	_Z14plain_direct32PK14__hip_fp8_e4m3S1_Pfiii@GOTPCREL(%rip), %rdi
	movq	%r15, %r9
	pushq	168(%rsp)
	.cfi_adjust_cfa_offset 8
	pushq	72(%rsp)
	.cfi_adjust_cfa_offset 8
	callq	hipLaunchKernel@PLT
	addq	$16, %rsp
	.cfi_adjust_cfa_offset -16
.Ltmp32:                                # EH_LABEL
	jmp	.LBB8_55
	.p2align	4
.LBB8_51:                               #   in Loop: Header=BB8_41 Depth=1
.Ltmp33:                                # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movq	%r14, %rdi
	movl	$1, %esi
	movq	%rbx, %rdx
	movl	$1, %ecx
	xorl	%r8d, %r8d
	xorl	%r9d, %r9d
	callq	__hipPushCallConfiguration@PLT
.Ltmp34:                                # EH_LABEL
# %bb.52:                               #   in Loop: Header=BB8_41 Depth=1
	testl	%eax, %eax
	jne	.LBB8_55
# %bb.53:                               #   in Loop: Header=BB8_41 Depth=1
	movq	160(%rsp), %rax
	movq	152(%rsp), %rcx
	movq	144(%rsp), %rdx
	movq	%rax, 232(%rsp)
	movq	%rcx, 112(%rsp)
	movq	%rdx, 104(%rsp)
	movq	48(%rsp), %rax                  # 8-byte Reload
	movl	%eax, 56(%rsp)
	movq	40(%rsp), %rax                  # 8-byte Reload
	movl	%eax, 24(%rsp)
	movl	16(%rsp), %eax                  # 4-byte Reload
	movl	%eax, 12(%rsp)
	movq	%r13, 176(%rsp)
	leaq	112(%rsp), %rax
	movq	%rax, 184(%rsp)
	leaq	104(%rsp), %rax
	movq	%rax, 192(%rsp)
	leaq	56(%rsp), %rax
	movq	%rax, 200(%rsp)
	leaq	24(%rsp), %rax
	movq	%rax, 208(%rsp)
	leaq	12(%rsp), %rax
	movq	%rax, 216(%rsp)
.Ltmp35:                                # EH_LABEL
	.cfi_escape 0x2e, 0x00
	leaq	88(%rsp), %rdi
	leaq	72(%rsp), %rsi
	leaq	64(%rsp), %rdx
	leaq	168(%rsp), %rcx
	callq	__hipPopCallConfiguration@PLT
.Ltmp36:                                # EH_LABEL
# %bb.54:                               # %.noexc117
                                        #   in Loop: Header=BB8_41 Depth=1
	movq	88(%rsp), %rsi
	movl	96(%rsp), %edx
	movq	72(%rsp), %rcx
	movl	80(%rsp), %r8d
.Ltmp37:                                # EH_LABEL
	.cfi_escape 0x2e, 0x10
	movq	_Z14plain_direct16PK14__hip_fp8_e4m3S1_Pfiii@GOTPCREL(%rip), %rdi
	movq	%r15, %r9
	pushq	168(%rsp)
	.cfi_adjust_cfa_offset 8
	pushq	72(%rsp)
	.cfi_adjust_cfa_offset 8
	callq	hipLaunchKernel@PLT
	addq	$16, %rsp
	.cfi_adjust_cfa_offset -16
.Ltmp38:                                # EH_LABEL
	.p2align	4
.LBB8_55:                               #   in Loop: Header=BB8_41 Depth=1
.Ltmp39:                                # EH_LABEL
	.cfi_escape 0x2e, 0x00
	callq	hipDeviceSynchronize@PLT
.Ltmp40:                                # EH_LABEL
# %bb.56:                               #   in Loop: Header=BB8_41 Depth=1
	decl	%ebp
	jne	.LBB8_41
# %bb.57:
.Ltmp42:                                # EH_LABEL
	.cfi_escape 0x2e, 0x00
	leaq	56(%rsp), %rdi
	callq	hipEventCreate@PLT
.Ltmp43:                                # EH_LABEL
# %bb.58:
.Ltmp44:                                # EH_LABEL
	.cfi_escape 0x2e, 0x00
	leaq	24(%rsp), %rdi
	callq	hipEventCreate@PLT
.Ltmp45:                                # EH_LABEL
# %bb.59:
	movq	56(%rsp), %rdi
.Ltmp46:                                # EH_LABEL
	.cfi_escape 0x2e, 0x00
	xorl	%esi, %esi
	callq	hipEventRecord@PLT
.Ltmp47:                                # EH_LABEL
# %bb.60:                               # %.preheader
	cmpl	$0, 124(%rsp)                   # 4-byte Folded Reload
	jle	.LBB8_73
# %bb.61:
	movabsq	$4294967328, %rbx               # imm = 0x100000020
	leaq	168(%rsp), %r15
	leaq	176(%rsp), %r12
	movl	124(%rsp), %ebp                 # 4-byte Reload
	jmp	.LBB8_62
	.p2align	4
.LBB8_72:                               #   in Loop: Header=BB8_62 Depth=1
	decl	%ebp
	je	.LBB8_73
.LBB8_62:                               # =>This Inner Loop Header: Depth=1
	cmpl	$0, 20(%rsp)                    # 4-byte Folded Reload
	je	.LBB8_68
# %bb.63:                               #   in Loop: Header=BB8_62 Depth=1
.Ltmp48:                                # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movq	%r14, %rdi
	movl	$1, %esi
	movq	%rbx, %rdx
	movl	$1, %ecx
	xorl	%r8d, %r8d
	xorl	%r9d, %r9d
	callq	__hipPushCallConfiguration@PLT
.Ltmp49:                                # EH_LABEL
# %bb.64:                               #   in Loop: Header=BB8_62 Depth=1
	testl	%eax, %eax
	jne	.LBB8_72
# %bb.65:                               #   in Loop: Header=BB8_62 Depth=1
	movq	160(%rsp), %rax
	movq	152(%rsp), %rcx
	movq	144(%rsp), %rdx
	movq	%rax, 232(%rsp)
	movq	%rcx, 112(%rsp)
	movq	%rdx, 104(%rsp)
	movq	48(%rsp), %rax                  # 8-byte Reload
	movl	%eax, 12(%rsp)
	movq	40(%rsp), %rax                  # 8-byte Reload
	movl	%eax, 132(%rsp)
	movl	16(%rsp), %eax                  # 4-byte Reload
	movl	%eax, 128(%rsp)
	movq	%r13, 176(%rsp)
	leaq	112(%rsp), %rax
	movq	%rax, 184(%rsp)
	leaq	104(%rsp), %rax
	movq	%rax, 192(%rsp)
	leaq	12(%rsp), %rax
	movq	%rax, 200(%rsp)
	leaq	132(%rsp), %rax
	movq	%rax, 208(%rsp)
	leaq	128(%rsp), %rax
	movq	%rax, 216(%rsp)
.Ltmp50:                                # EH_LABEL
	.cfi_escape 0x2e, 0x00
	leaq	88(%rsp), %rdi
	leaq	72(%rsp), %rsi
	leaq	64(%rsp), %rdx
	movq	%r15, %rcx
	callq	__hipPopCallConfiguration@PLT
.Ltmp51:                                # EH_LABEL
# %bb.66:                               # %.noexc125
                                        #   in Loop: Header=BB8_62 Depth=1
	movq	88(%rsp), %rsi
	movl	96(%rsp), %edx
	movq	72(%rsp), %rcx
	movl	80(%rsp), %r8d
.Ltmp52:                                # EH_LABEL
	.cfi_escape 0x2e, 0x10
	movq	_Z14plain_direct32PK14__hip_fp8_e4m3S1_Pfiii@GOTPCREL(%rip), %rdi
	movq	%r12, %r9
	pushq	168(%rsp)
	.cfi_adjust_cfa_offset 8
	pushq	72(%rsp)
	.cfi_adjust_cfa_offset 8
	callq	hipLaunchKernel@PLT
	addq	$16, %rsp
	.cfi_adjust_cfa_offset -16
.Ltmp53:                                # EH_LABEL
	jmp	.LBB8_72
	.p2align	4
.LBB8_68:                               #   in Loop: Header=BB8_62 Depth=1
.Ltmp54:                                # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movq	%r14, %rdi
	movl	$1, %esi
	movq	%rbx, %rdx
	movl	$1, %ecx
	xorl	%r8d, %r8d
	xorl	%r9d, %r9d
	callq	__hipPushCallConfiguration@PLT
.Ltmp55:                                # EH_LABEL
# %bb.69:                               #   in Loop: Header=BB8_62 Depth=1
	testl	%eax, %eax
	jne	.LBB8_72
# %bb.70:                               #   in Loop: Header=BB8_62 Depth=1
	movq	160(%rsp), %rax
	movq	152(%rsp), %rcx
	movq	144(%rsp), %rdx
	movq	%rax, 232(%rsp)
	movq	%rcx, 112(%rsp)
	movq	%rdx, 104(%rsp)
	movq	48(%rsp), %rax                  # 8-byte Reload
	movl	%eax, 12(%rsp)
	movq	40(%rsp), %rax                  # 8-byte Reload
	movl	%eax, 132(%rsp)
	movl	16(%rsp), %eax                  # 4-byte Reload
	movl	%eax, 128(%rsp)
	movq	%r13, 176(%rsp)
	leaq	112(%rsp), %rax
	movq	%rax, 184(%rsp)
	leaq	104(%rsp), %rax
	movq	%rax, 192(%rsp)
	leaq	12(%rsp), %rax
	movq	%rax, 200(%rsp)
	leaq	132(%rsp), %rax
	movq	%rax, 208(%rsp)
	leaq	128(%rsp), %rax
	movq	%rax, 216(%rsp)
.Ltmp56:                                # EH_LABEL
	.cfi_escape 0x2e, 0x00
	leaq	88(%rsp), %rdi
	leaq	72(%rsp), %rsi
	leaq	64(%rsp), %rdx
	movq	%r15, %rcx
	callq	__hipPopCallConfiguration@PLT
.Ltmp57:                                # EH_LABEL
# %bb.71:                               # %.noexc134
                                        #   in Loop: Header=BB8_62 Depth=1
	movq	88(%rsp), %rsi
	movl	96(%rsp), %edx
	movq	72(%rsp), %rcx
	movl	80(%rsp), %r8d
.Ltmp58:                                # EH_LABEL
	.cfi_escape 0x2e, 0x10
	movq	_Z14plain_direct16PK14__hip_fp8_e4m3S1_Pfiii@GOTPCREL(%rip), %rdi
	movq	%r12, %r9
	pushq	168(%rsp)
	.cfi_adjust_cfa_offset 8
	pushq	72(%rsp)
	.cfi_adjust_cfa_offset 8
	callq	hipLaunchKernel@PLT
	addq	$16, %rsp
	.cfi_adjust_cfa_offset -16
.Ltmp59:                                # EH_LABEL
	jmp	.LBB8_72
.LBB8_73:                               # %._crit_edge192
	movq	24(%rsp), %rdi
.Ltmp61:                                # EH_LABEL
	.cfi_escape 0x2e, 0x00
	xorl	%esi, %esi
	callq	hipEventRecord@PLT
.Ltmp62:                                # EH_LABEL
# %bb.74:
	movq	24(%rsp), %rdi
.Ltmp63:                                # EH_LABEL
	.cfi_escape 0x2e, 0x00
	callq	hipEventSynchronize@PLT
.Ltmp64:                                # EH_LABEL
# %bb.75:
	movq	56(%rsp), %rsi
	movq	24(%rsp), %rdx
.Ltmp66:                                # EH_LABEL
	.cfi_escape 0x2e, 0x00
	leaq	176(%rsp), %rdi
	callq	hipEventElapsedTime@PLT
.Ltmp67:                                # EH_LABEL
# %bb.76:
	cvtsi2ssl	124(%rsp), %xmm0        # 4-byte Folded Reload
	movss	176(%rsp), %xmm1                # xmm1 = mem[0],zero,zero,zero
	divss	%xmm0, %xmm1
	movss	%xmm1, 176(%rsp)
	movq	160(%rsp), %rdi
.Ltmp68:                                # EH_LABEL
	.cfi_escape 0x2e, 0x00
	callq	hipFree@PLT
.Ltmp69:                                # EH_LABEL
# %bb.77:
	movq	152(%rsp), %rdi
.Ltmp70:                                # EH_LABEL
	.cfi_escape 0x2e, 0x00
	callq	hipFree@PLT
.Ltmp71:                                # EH_LABEL
# %bb.78:
	movq	144(%rsp), %rdi
.Ltmp72:                                # EH_LABEL
	.cfi_escape 0x2e, 0x00
	callq	hipFree@PLT
.Ltmp73:                                # EH_LABEL
# %bb.79:
	movq	56(%rsp), %rdi
.Ltmp74:                                # EH_LABEL
	.cfi_escape 0x2e, 0x00
	callq	hipEventDestroy@PLT
.Ltmp75:                                # EH_LABEL
# %bb.80:
	movq	24(%rsp), %rdi
.Ltmp76:                                # EH_LABEL
	.cfi_escape 0x2e, 0x00
	callq	hipEventDestroy@PLT
.Ltmp77:                                # EH_LABEL
# %bb.81:
	movss	176(%rsp), %xmm0                # xmm0 = mem[0],zero,zero,zero
	movss	%xmm0, 20(%rsp)                 # 4-byte Spill
	movq	136(%rsp), %rdi                 # 8-byte Reload
	testq	%rdi, %rdi
	movq	248(%rsp), %rbx                 # 8-byte Reload
	movq	32(%rsp), %r14                  # 8-byte Reload
	movq	240(%rsp), %rsi                 # 8-byte Reload
	je	.LBB8_83
# %bb.82:
	subq	%rdi, %rsi
	.cfi_escape 0x2e, 0x00
	callq	_ZdlPvm@PLT
.LBB8_83:                               # %_ZNSt6vectorI14__hip_fp8_e4m3SaIS0_EED2Ev.exit
	testq	%r14, %r14
	je	.LBB8_85
# %bb.84:
	subq	%r14, %rbx
	.cfi_escape 0x2e, 0x00
	movq	%r14, %rdi
	movq	%rbx, %rsi
	callq	_ZdlPvm@PLT
.LBB8_85:                               # %_ZNSt6vectorI14__hip_fp8_e4m3SaIS0_EED2Ev.exit138
	movss	20(%rsp), %xmm0                 # 4-byte Reload
                                        # xmm0 = mem[0],zero,zero,zero
	cvtss2sd	%xmm0, %xmm0
	addq	$5272, %rsp                     # imm = 0x1498
	.cfi_def_cfa_offset 56
	popq	%rbx
	.cfi_def_cfa_offset 48
	popq	%r12
	.cfi_def_cfa_offset 40
	popq	%r13
	.cfi_def_cfa_offset 32
	popq	%r14
	.cfi_def_cfa_offset 24
	popq	%r15
	.cfi_def_cfa_offset 16
	popq	%rbp
	.cfi_def_cfa_offset 8
	retq
.LBB8_93:                               # %.noexc
	.cfi_def_cfa_offset 5328
	.cfi_escape 0x2e, 0x00
	leaq	.L.str.55(%rip), %rdi
	callq	_ZSt20__throw_length_errorPKc@PLT
.LBB8_9:
.Ltmp79:                                # EH_LABEL
	.cfi_escape 0x2e, 0x00
	leaq	.L.str.55(%rip), %rdi
	callq	_ZSt20__throw_length_errorPKc@PLT
.Ltmp80:                                # EH_LABEL
# %bb.10:                               # %.noexc102
.LBB8_26:
.Ltmp81:                                # EH_LABEL
	movq	%rax, %rbx
	testq	%rbp, %rbp
	je	.LBB8_92
	jmp	.LBB8_91
.LBB8_94:
.Ltmp65:                                # EH_LABEL
	jmp	.LBB8_87
.LBB8_49:
.Ltmp24:                                # EH_LABEL
	movq	%rax, %rbx
	jmp	.LBB8_88
.LBB8_86:
.Ltmp78:                                # EH_LABEL
	jmp	.LBB8_87
.LBB8_47:
.Ltmp10:                                # EH_LABEL
	movq	%rax, %rbx
	jmp	.LBB8_88
.LBB8_48:
.Ltmp13:                                # EH_LABEL
	movq	%rax, %rbx
	jmp	.LBB8_88
.LBB8_30:
.Ltmp4:                                 # EH_LABEL
	jmp	.LBB8_87
.LBB8_31:
.Ltmp7:                                 # EH_LABEL
	jmp	.LBB8_87
.LBB8_67:
.Ltmp60:                                # EH_LABEL
	jmp	.LBB8_87
.LBB8_50:
.Ltmp41:                                # EH_LABEL
.LBB8_87:
	movq	%rax, %rbx
	movq	32(%rsp), %rbp                  # 8-byte Reload
.LBB8_88:
	movq	136(%rsp), %rdi                 # 8-byte Reload
	testq	%rdi, %rdi
	jne	.LBB8_89
# %bb.90:                               # %_ZNSt6vectorI14__hip_fp8_e4m3SaIS0_EED2Ev.exit140
	testq	%rbp, %rbp
	jne	.LBB8_91
.LBB8_92:                               # %_ZNSt6vectorI14__hip_fp8_e4m3SaIS0_EED2Ev.exit142
	.cfi_escape 0x2e, 0x00
	movq	%rbx, %rdi
	callq	_Unwind_Resume@PLT
.LBB8_89:
	movq	240(%rsp), %rsi                 # 8-byte Reload
	subq	%rdi, %rsi
	.cfi_escape 0x2e, 0x00
	callq	_ZdlPvm@PLT
	testq	%rbp, %rbp
	je	.LBB8_92
.LBB8_91:
	movq	248(%rsp), %rsi                 # 8-byte Reload
	subq	%rbp, %rsi
	.cfi_escape 0x2e, 0x00
	movq	%rbp, %rdi
	callq	_ZdlPvm@PLT
	.cfi_escape 0x2e, 0x00
	movq	%rbx, %rdi
	callq	_Unwind_Resume@PLT
.Lfunc_end8:
	.size	_Z11bench_plainiiiii, .Lfunc_end8-_Z11bench_plainiiiii
	.cfi_endproc
	.section	.gcc_except_table,"a",@progbits
	.p2align	2, 0x0
GCC_except_table8:
.Lexception0:
	.byte	255                             # @LPStart Encoding = omit
	.byte	255                             # @TType Encoding = omit
	.byte	1                               # Call site Encoding = uleb128
	.uleb128 .Lcst_end0-.Lcst_begin0
.Lcst_begin0:
	.uleb128 .Lfunc_begin0-.Lfunc_begin0    # >> Call Site 1 <<
	.uleb128 .Ltmp0-.Lfunc_begin0           #   Call between .Lfunc_begin0 and .Ltmp0
	.byte	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp0-.Lfunc_begin0           # >> Call Site 2 <<
	.uleb128 .Ltmp1-.Ltmp0                  #   Call between .Ltmp0 and .Ltmp1
	.uleb128 .Ltmp81-.Lfunc_begin0          #     jumps to .Ltmp81
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp1-.Lfunc_begin0           # >> Call Site 3 <<
	.uleb128 .Ltmp2-.Ltmp1                  #   Call between .Ltmp1 and .Ltmp2
	.byte	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp2-.Lfunc_begin0           # >> Call Site 4 <<
	.uleb128 .Ltmp3-.Ltmp2                  #   Call between .Ltmp2 and .Ltmp3
	.uleb128 .Ltmp4-.Lfunc_begin0           #     jumps to .Ltmp4
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp5-.Lfunc_begin0           # >> Call Site 5 <<
	.uleb128 .Ltmp6-.Ltmp5                  #   Call between .Ltmp5 and .Ltmp6
	.uleb128 .Ltmp7-.Lfunc_begin0           #     jumps to .Ltmp7
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp8-.Lfunc_begin0           # >> Call Site 6 <<
	.uleb128 .Ltmp9-.Ltmp8                  #   Call between .Ltmp8 and .Ltmp9
	.uleb128 .Ltmp10-.Lfunc_begin0          #     jumps to .Ltmp10
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp11-.Lfunc_begin0          # >> Call Site 7 <<
	.uleb128 .Ltmp12-.Ltmp11                #   Call between .Ltmp11 and .Ltmp12
	.uleb128 .Ltmp13-.Lfunc_begin0          #     jumps to .Ltmp13
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp14-.Lfunc_begin0          # >> Call Site 8 <<
	.uleb128 .Ltmp23-.Ltmp14                #   Call between .Ltmp14 and .Ltmp23
	.uleb128 .Ltmp24-.Lfunc_begin0          #     jumps to .Ltmp24
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp25-.Lfunc_begin0          # >> Call Site 9 <<
	.uleb128 .Ltmp40-.Ltmp25                #   Call between .Ltmp25 and .Ltmp40
	.uleb128 .Ltmp41-.Lfunc_begin0          #     jumps to .Ltmp41
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp42-.Lfunc_begin0          # >> Call Site 10 <<
	.uleb128 .Ltmp47-.Ltmp42                #   Call between .Ltmp42 and .Ltmp47
	.uleb128 .Ltmp65-.Lfunc_begin0          #     jumps to .Ltmp65
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp48-.Lfunc_begin0          # >> Call Site 11 <<
	.uleb128 .Ltmp59-.Ltmp48                #   Call between .Ltmp48 and .Ltmp59
	.uleb128 .Ltmp60-.Lfunc_begin0          #     jumps to .Ltmp60
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp61-.Lfunc_begin0          # >> Call Site 12 <<
	.uleb128 .Ltmp64-.Ltmp61                #   Call between .Ltmp61 and .Ltmp64
	.uleb128 .Ltmp65-.Lfunc_begin0          #     jumps to .Ltmp65
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp66-.Lfunc_begin0          # >> Call Site 13 <<
	.uleb128 .Ltmp77-.Ltmp66                #   Call between .Ltmp66 and .Ltmp77
	.uleb128 .Ltmp78-.Lfunc_begin0          #     jumps to .Ltmp78
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp77-.Lfunc_begin0          # >> Call Site 14 <<
	.uleb128 .Ltmp79-.Ltmp77                #   Call between .Ltmp77 and .Ltmp79
	.byte	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp79-.Lfunc_begin0          # >> Call Site 15 <<
	.uleb128 .Ltmp80-.Ltmp79                #   Call between .Ltmp79 and .Ltmp80
	.uleb128 .Ltmp81-.Lfunc_begin0          #     jumps to .Ltmp81
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp80-.Lfunc_begin0          # >> Call Site 16 <<
	.uleb128 .Lfunc_end8-.Ltmp80            #   Call between .Ltmp80 and .Lfunc_end8
	.byte	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
.Lcst_end0:
	.p2align	2, 0x0
                                        # -- End function
	.section	.rodata.cst16,"aM",@progbits,16
	.p2align	4, 0x0                          # -- Begin function _ZNSt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EEclEv
.LCPI9_0:
	.quad	-2147483648                     # 0xffffffff80000000
	.quad	-2147483648                     # 0xffffffff80000000
.LCPI9_1:
	.quad	2147483646                      # 0x7ffffffe
	.quad	2147483646                      # 0x7ffffffe
.LCPI9_2:
	.quad	2567483615                      # 0x9908b0df
	.quad	2567483615                      # 0x9908b0df
	.section	.text._ZNSt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EEclEv,"axG",@progbits,_ZNSt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EEclEv,comdat
	.weak	_ZNSt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EEclEv
	.p2align	1
	.prefalign	4, .Lfunc_end9, nop
	.type	_ZNSt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EEclEv,@function
_ZNSt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EEclEv: # @_ZNSt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EEclEv
	.cfi_startproc
# %bb.0:
	movq	4992(%rdi), %rax
	cmpq	$624, %rax                      # imm = 0x270
	jb	.LBB9_6
# %bb.1:                                # %vector.ph
	movq	(%rdi), %xmm0                   # xmm0 = mem[0],zero
	pshufd	$68, %xmm0, %xmm3               # xmm3 = xmm0[0,1,0,1]
	xorl	%eax, %eax
	movaps	.LCPI9_0(%rip), %xmm0           # xmm0 = [18446744071562067968,18446744071562067968]
	movaps	.LCPI9_1(%rip), %xmm1           # xmm1 = [2147483646,2147483646]
	movdqa	.LCPI9_2(%rip), %xmm2           # xmm2 = [2567483615,2567483615]
	.p2align	4
.LBB9_2:                                # %vector.body
                                        # =>This Inner Loop Header: Depth=1
	movdqa	%xmm3, %xmm4
	movups	8(%rdi,%rax,8), %xmm3
	shufps	$78, %xmm3, %xmm4               # xmm4 = xmm4[2,3],xmm3[0,1]
	andps	%xmm0, %xmm4
	movaps	%xmm3, %xmm5
	andps	%xmm1, %xmm5
	orps	%xmm4, %xmm5
	movdqu	3176(%rdi,%rax,8), %xmm4
	psrlq	$1, %xmm5
	pxor	%xmm4, %xmm5
	pshufd	$160, %xmm3, %xmm4              # xmm4 = xmm3[0,0,2,2]
	pslld	$31, %xmm4
	psrad	$31, %xmm4
	pand	%xmm2, %xmm4
	pxor	%xmm5, %xmm4
	movdqu	%xmm4, (%rdi,%rax,8)
	addq	$2, %rax
	cmpq	$226, %rax
	jne	.LBB9_2
# %bb.3:                                # %vector.ph11
	movl	$2567483615, %eax               # imm = 0x9908B0DF
	pshufd	$238, %xmm3, %xmm3              # xmm3 = xmm3[2,3,2,3]
	movq	%xmm3, %rcx
	andq	$-2147483648, %rcx              # imm = 0x80000000
	movq	1816(%rdi), %rdx
	movl	%edx, %esi
	movq	%rdx, %xmm3
                                        # kill: def $edx killed $edx killed $rdx def $rdx
	andl	$2147483646, %edx               # imm = 0x7FFFFFFE
	orq	%rcx, %rdx
	shrq	%rdx
	xorq	4984(%rdi), %rdx
	andl	$1, %esi
	negl	%esi
	movl	$2567483615, %ecx               # imm = 0x9908B0DF
	andl	%esi, %ecx
	xorq	%rdx, %rcx
	movq	%rcx, 1808(%rdi)
	pshufd	$68, %xmm3, %xmm3               # xmm3 = xmm3[0,1,0,1]
	movl	$228, %ecx
	.p2align	4
.LBB9_4:                                # %vector.body12
                                        # =>This Inner Loop Header: Depth=1
	movups	(%rdi,%rcx,8), %xmm4
	shufps	$78, %xmm4, %xmm3               # xmm3 = xmm3[2,3],xmm4[0,1]
	andps	%xmm0, %xmm3
	movaps	%xmm4, %xmm5
	andps	%xmm1, %xmm5
	orps	%xmm3, %xmm5
	movdqu	-1824(%rdi,%rcx,8), %xmm3
	psrlq	$1, %xmm5
	pxor	%xmm3, %xmm5
	pshufd	$160, %xmm4, %xmm3              # xmm3 = xmm4[0,0,2,2]
	pslld	$31, %xmm3
	psrad	$31, %xmm3
	pand	%xmm2, %xmm3
	pxor	%xmm5, %xmm3
	movdqu	%xmm3, -8(%rdi,%rcx,8)
	addq	$2, %rcx
	movdqa	%xmm4, %xmm3
	cmpq	$624, %rcx                      # imm = 0x270
	jne	.LBB9_4
# %bb.5:                                # %_ZNSt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EE11_M_gen_randEv.exit
	movq	$-2147483648, %rcx              # imm = 0x80000000
	andq	4984(%rdi), %rcx
	movq	(%rdi), %rdx
	movl	%edx, %esi
	andl	$2147483646, %esi               # imm = 0x7FFFFFFE
	orq	%rcx, %rsi
	shrq	%rsi
	xorq	3168(%rdi), %rsi
	andl	$1, %edx
	negl	%edx
	andl	%eax, %edx
	xorq	%rsi, %rdx
	movq	%rdx, 4984(%rdi)
	xorl	%eax, %eax
.LBB9_6:
	leaq	1(%rax), %rcx
	movq	%rcx, 4992(%rdi)
	movq	(%rdi,%rax,8), %rax
	movq	%rax, %rcx
	shrq	$11, %rcx
	movl	%ecx, %ecx
	xorq	%rax, %rcx
	movl	%ecx, %eax
	shll	$7, %eax
	andl	$-1658038656, %eax              # imm = 0x9D2C5680
	xorq	%rcx, %rax
	movl	%eax, %ecx
	shll	$15, %ecx
	andl	$-272236544, %ecx               # imm = 0xEFC60000
	xorq	%rax, %rcx
	movq	%rcx, %rax
	shrq	$18, %rax
	xorq	%rcx, %rax
	retq
.Lfunc_end9:
	.size	_ZNSt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EEclEv, .Lfunc_end9-_ZNSt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EEclEv
	.cfi_endproc
                                        # -- End function
	.section	.text._ZN14__hip_fp8_e4m3C2Ef,"axG",@progbits,_ZN14__hip_fp8_e4m3C2Ef,comdat
	.weak	_ZN14__hip_fp8_e4m3C2Ef         # -- Begin function _ZN14__hip_fp8_e4m3C2Ef
	.p2align	1
	.prefalign	4, .Lfunc_end10, nop
	.type	_ZN14__hip_fp8_e4m3C2Ef,@function
_ZN14__hip_fp8_e4m3C2Ef:                # @_ZN14__hip_fp8_e4m3C2Ef
	.cfi_startproc
# %bb.0:
	movd	%xmm0, %eax
	movl	%eax, %ecx
	notl	%ecx
	movl	%eax, %edx
	shrl	$24, %edx
	testl	$2139095040, %ecx               # imm = 0x7F800000
	jne	.LBB10_2
# %bb.1:
	orb	$127, %dl
	movb	%dl, (%rdi)
	retq
.LBB10_2:
	andl	$-128, %edx
	movl	%eax, %ecx
	andl	$2147483647, %ecx               # imm = 0x7FFFFFFF
	cmpl	$1138753537, %ecx               # imm = 0x43E00001
	jb	.LBB10_4
# %bb.3:
	orb	$126, %dl
	movb	%dl, (%rdi)
	retq
.LBB10_4:
	testq	%rax, %rax
	je	.LBB10_5
# %bb.6:
	movl	%eax, %esi
	andl	$8388607, %esi                  # imm = 0x7FFFFF
	shrl	$23, %eax
	movzbl	%al, %ecx
	testl	%ecx, %ecx
	je	.LBB10_7
# %bb.8:
	leal	-127(%rcx), %r8d
	movl	$121, %r9d
	subl	%ecx, %r9d
	orq	$8388608, %rsi                  # imm = 0x800000
	xorl	%eax, %eax
	cmpl	$122, %ecx
	cmovbl	%r9d, %eax
	jmp	.LBB10_9
.LBB10_5:
	xorl	%edx, %edx
	movb	%dl, (%rdi)
	retq
.LBB10_7:
	movl	$120, %eax
	movl	$-126, %r8d
.LBB10_9:                               # %select.unfold.i.i
	leal	20(%rax), %ecx
	movq	$-1, %r10
                                        # kill: def $cl killed $cl killed $ecx
	shlq	%cl, %r10
	notl	%r10d
	andl	%esi, %r10d
	leal	19(%rax), %ecx
	movl	$1, %r11d
                                        # kill: def $cl killed $cl killed $ecx
	shlq	%cl, %r11
	movl	%eax, %ecx
	shrq	%cl, %rsi
	addl	%eax, %r8d
	movl	%esi, %r9d
	shrl	$23, %r9d
	addl	%r8d, %r9d
	xorl	%ecx, %ecx
	btl	$20, %esi
	movl	$0, %eax
	adcl	$1048575, %eax                  # imm = 0xFFFFF
	cmpq	%r11, %r10
	cmovnel	%ecx, %eax
	addl	%esi, %eax
	andl	$1048575, %eax                  # imm = 0xFFFFF
	addq	%rsi, %rax
	movl	%r9d, %ecx
	addl	$6, %ecx
	je	.LBB10_10
# %bb.11:
	testl	$16777216, %eax                 # imm = 0x1000000
	je	.LBB10_13
# %bb.12:
	shrl	%eax
	addl	$7, %r9d
	movl	%r9d, %ecx
	jmp	.LBB10_13
.LBB10_10:
	movl	%eax, %ecx
	shrl	$23, %ecx
	andl	$1, %ecx
.LBB10_13:
	shrl	$20, %eax
	cmpl	$15, %ecx
	movl	$15, %esi
	cmovll	%ecx, %esi
	movl	%eax, %eax
	movl	$7, %r8d
	cmovleq	%rax, %r8
	movl	%r8d, %eax
	andl	$7, %eax
	shll	$3, %esi
	orl	%edx, %esi
	orl	%eax, %esi
	testq	%r8, %r8
	cmovnel	%esi, %edx
	testl	%ecx, %ecx
	cmovnel	%esi, %edx
	movb	%dl, (%rdi)
	retq
.Lfunc_end10:
	.size	_ZN14__hip_fp8_e4m3C2Ef, .Lfunc_end10-_ZN14__hip_fp8_e4m3C2Ef
	.cfi_endproc
                                        # -- End function
	.section	.rodata.cst4,"aM",@progbits,4
	.p2align	2, 0x0                          # -- Begin function _Z14bench_fused_m5iiii
.LCPI11_0:
	.long	0x3f000000                      # float 0.5
.LCPI11_1:
	.long	0x41200000                      # float 10
	.text
	.globl	_Z14bench_fused_m5iiii
	.prefalign	4, .Lfunc_end11, nop
	.type	_Z14bench_fused_m5iiii,@function
_Z14bench_fused_m5iiii:                 # @_Z14bench_fused_m5iiii
.Lfunc_begin1:
	.cfi_startproc
	.cfi_personality 155, DW.ref.__gxx_personality_v0
	.cfi_lsda 27, .Lexception1
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	pushq	%r15
	.cfi_def_cfa_offset 24
	pushq	%r14
	.cfi_def_cfa_offset 32
	pushq	%r13
	.cfi_def_cfa_offset 40
	pushq	%r12
	.cfi_def_cfa_offset 48
	pushq	%rbx
	.cfi_def_cfa_offset 56
	subq	$5304, %rsp                     # imm = 0x14B8
	.cfi_def_cfa_offset 5360
	.cfi_offset %rbx, -56
	.cfi_offset %r12, -48
	.cfi_offset %r13, -40
	.cfi_offset %r14, -32
	.cfi_offset %r15, -24
	.cfi_offset %rbp, -16
	movl	%ecx, 20(%rsp)                  # 4-byte Spill
                                        # kill: def $edx killed $edx def $rdx
                                        # kill: def $esi killed $esi def $rsi
	movq	%rsi, 8(%rsp)                   # 8-byte Spill
                                        # kill: def $edi killed $edi def $rdi
	movq	%rdi, 48(%rsp)                  # 8-byte Spill
	leal	255(%rdx), %esi
	testl	%edx, %edx
	movq	%rdx, 128(%rsp)                 # 8-byte Spill
	cmovnsl	%edx, %esi
	sarl	$8, %esi
	movq	$2, 304(%rsp)
	movl	$2, %ecx
	movl	$2, %eax
	.p2align	4
.LBB11_1:                               # =>This Inner Loop Header: Depth=1
	movq	%rcx, %rdx
	shrq	$30, %rdx
	xorq	%rcx, %rdx
	imulq	$1812433253, %rdx, %rcx         # imm = 0x6C078965
	addq	%rax, %rcx
	decq	%rcx
	movl	%ecx, %edx
	movq	%rdx, 296(%rsp,%rax,8)
	cmpq	$624, %rax                      # imm = 0x270
	je	.LBB11_3
# %bb.2:                                #   in Loop: Header=BB11_1 Depth=1
	shrl	$30, %edx
	xorl	%edx, %ecx
	imull	$1812433253, %ecx, %ecx         # imm = 0x6C078965
	addl	%eax, %ecx
	movq	%rcx, 304(%rsp,%rax,8)
	addq	$2, %rax
	jmp	.LBB11_1
.LBB11_3:                               # %_ZNSt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EEC2Em.exit
	movl	%esi, %eax
	shll	$7, %eax
	movq	$624, 5296(%rsp)                # imm = 0x270
	movl	%eax, 68(%rsp)                  # 4-byte Spill
	imull	8(%rsp), %eax                   # 4-byte Folded Reload
	testl	%eax, %eax
	js	.LBB11_84
# %bb.4:                                # %_ZNSt6vectorIhSaIhEE17_S_check_init_lenEmRKS0_.exit.i
	movslq	%eax, %rbx
	je	.LBB11_5
# %bb.14:                               # %_ZNSt6vectorIhSaIhEEC2EmRKS0_.exit
	.cfi_escape 0x2e, 0x00
	movq	%rbx, %rdi
	callq	_Znwm@PLT
	leaq	(%rax,%rbx), %rcx
	movq	%rcx, 120(%rsp)                 # 8-byte Spill
	movb	$0, (%rax)
	movq	%rax, 32(%rsp)                  # 8-byte Spill
	leaq	1(%rax), %rdi
	leaq	-1(%rbx), %rdx
	.cfi_escape 0x2e, 0x00
	xorl	%r15d, %r15d
	xorl	%esi, %esi
	callq	memset@PLT
	leaq	304(%rsp), %r14
	.p2align	4
.LBB11_15:                              # %.lr.ph
                                        # =>This Inner Loop Header: Depth=1
.Ltmp82:                                # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movq	%r14, %rdi
	callq	_ZNSt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EEclEv
.Ltmp83:                                # EH_LABEL
# %bb.16:                               #   in Loop: Header=BB11_15 Depth=1
	movq	32(%rsp), %rcx                  # 8-byte Reload
	movb	%al, (%rcx,%r15)
	incq	%r15
	cmpq	%r15, %rbx
	jne	.LBB11_15
	jmp	.LBB11_6
.LBB11_5:
	movq	$0, 32(%rsp)                    # 8-byte Folded Spill
	movq	$0, 120(%rsp)                   # 8-byte Folded Spill
.LBB11_6:                               # %._crit_edge
.Ltmp85:                                # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movl	$2048, %edi                     # imm = 0x800
	callq	_Znwm@PLT
.Ltmp86:                                # EH_LABEL
# %bb.7:                                # %.lr.ph194.preheader
	.cfi_escape 0x2e, 0x00
	xorl	%r15d, %r15d
	movl	$2048, %edx                     # imm = 0x800
	movq	%rax, 72(%rsp)                  # 8-byte Spill
	movq	%rax, %rdi
	xorl	%esi, %esi
	callq	memset@PLT
	leaq	304(%rsp), %r14
	movabsq	$-3689348814741910323, %r12     # imm = 0xCCCCCCCCCCCCCCCD
	jmp	.LBB11_8
	.p2align	4
.LBB11_11:                              #   in Loop: Header=BB11_8 Depth=1
	incq	%r15
	cmpq	$2048, %r15                     # imm = 0x800
	je	.LBB11_12
.LBB11_8:                               # %.lr.ph194
                                        # =>This Inner Loop Header: Depth=1
.Ltmp88:                                # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movq	%r14, %rdi
	callq	_ZNSt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EEclEv
.Ltmp89:                                # EH_LABEL
# %bb.9:                                #   in Loop: Header=BB11_8 Depth=1
	movq	%rax, %rcx
	mulq	%r12
	shrq	$2, %rdx
	leal	(%rdx,%rdx,4), %eax
	subl	%eax, %ecx
	xorps	%xmm0, %xmm0
	cvtsi2ss	%ecx, %xmm0
	mulss	.LCPI11_0(%rip), %xmm0
	mulss	.LCPI11_1(%rip), %xmm0
	cvttss2si	%xmm0, %eax
	movq	72(%rsp), %rcx                  # 8-byte Reload
	movb	%al, (%rcx,%r15)
	orl	$128, %eax
	movzbl	%al, %eax
	cmpl	$255, %eax
	jne	.LBB11_11
# %bb.10:                               #   in Loop: Header=BB11_8 Depth=1
	movb	$0, (%rcx,%r15)
	jmp	.LBB11_11
.LBB11_12:                              # %._crit_edge195
	movq	%rbx, 296(%rsp)                 # 8-byte Spill
	movq	128(%rsp), %rax                 # 8-byte Reload
                                        # kill: def $eax killed $eax killed $rax
	imull	48(%rsp), %eax                  # 4-byte Folded Reload
	testl	%eax, %eax
	js	.LBB11_13
# %bb.19:                               # %_ZNSt6vectorI14__hip_fp8_e4m3SaIS0_EE17_S_check_init_lenEmRKS1_.exit.i
	movslq	%eax, %r13
	je	.LBB11_20
# %bb.60:
.Ltmp91:                                # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movq	%r13, %rdi
	callq	_Znwm@PLT
.Ltmp92:                                # EH_LABEL
# %bb.61:                               # %.noexc119
	movq	%rax, %r14
	movb	$0, (%rax)
	movq	%r13, %rdx
	decq	%rdx
	je	.LBB11_63
# %bb.62:                               # %_ZNSt6vectorI14__hip_fp8_e4m3SaIS0_EEC2EmRKS1_.exit
	leaq	1(%r14), %rdi
	.cfi_escape 0x2e, 0x00
	xorl	%esi, %esi
	callq	memset@PLT
.LBB11_63:                              # %.lr.ph198.preheader
	movq	%r14, %r15
	addq	%r13, %r14
	movq	%r14, 112(%rsp)                 # 8-byte Spill
	xorl	%ebp, %ebp
	leaq	304(%rsp), %r14
	movabsq	$-2049638230412172401, %rbx     # imm = 0xE38E38E38E38E38F
	leaq	224(%rsp), %r12
	.p2align	4
.LBB11_64:                              # %.lr.ph198
                                        # =>This Inner Loop Header: Depth=1
.Ltmp93:                                # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movq	%r14, %rdi
	callq	_ZNSt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EEclEv
.Ltmp94:                                # EH_LABEL
# %bb.65:                               #   in Loop: Header=BB11_64 Depth=1
	movq	%rax, %rcx
	mulq	%rbx
	shrq	$3, %rdx
	leaq	(%rdx,%rdx,8), %rax
	negq	%rax
	addq	%rcx, %rax
	addq	$-4, %rax
	testq	%rax, %rax
	js	.LBB11_66
# %bb.70:                               #   in Loop: Header=BB11_64 Depth=1
	xorps	%xmm0, %xmm0
	cvtsi2ss	%rax, %xmm0
	jmp	.LBB11_71
	.p2align	4
.LBB11_66:                              #   in Loop: Header=BB11_64 Depth=1
	movq	%rax, %rcx
	shrq	%rcx
	andl	$1, %eax
	orq	%rcx, %rax
	xorps	%xmm0, %xmm0
	cvtsi2ss	%rax, %xmm0
	addss	%xmm0, %xmm0
.LBB11_71:                              #   in Loop: Header=BB11_64 Depth=1
	mulss	.LCPI11_0(%rip), %xmm0
.Ltmp96:                                # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movq	%r12, %rdi
	callq	_ZN14__hip_fp8_e4m3C2Ef
.Ltmp97:                                # EH_LABEL
# %bb.72:                               #   in Loop: Header=BB11_64 Depth=1
	movzbl	224(%rsp), %eax
	movb	%al, (%r15,%rbp)
	incq	%rbp
	cmpq	%rbp, %r13
	jne	.LBB11_64
	jmp	.LBB11_21
.LBB11_20:
	xorl	%r15d, %r15d
	movq	$0, 112(%rsp)                   # 8-byte Folded Spill
.LBB11_21:                              # %._crit_edge199
.Ltmp99:                                # EH_LABEL
	.cfi_escape 0x2e, 0x00
	leaq	104(%rsp), %rdi
	movq	%r13, %rsi
	callq	hipMalloc@PLT
.Ltmp100:                               # EH_LABEL
# %bb.22:                               # %_ZL9hipMallocI14__hip_fp8_e4m3E10hipError_tPPT_m.exit
	movq	8(%rsp), %rax                   # 8-byte Reload
                                        # kill: def $eax killed $eax killed $rax
	imull	48(%rsp), %eax                  # 4-byte Folded Reload
	movslq	%eax, %r12
	shlq	$2, %r12
.Ltmp101:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	leaq	96(%rsp), %rdi
	movq	%r12, %rsi
	callq	hipMalloc@PLT
.Ltmp102:                               # EH_LABEL
# %bb.23:                               # %_ZL9hipMallocIfE10hipError_tPPT_m.exit
.Ltmp103:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	leaq	88(%rsp), %rdi
	movq	296(%rsp), %rsi                 # 8-byte Reload
	callq	hipMalloc@PLT
.Ltmp104:                               # EH_LABEL
# %bb.24:                               # %_ZL9hipMallocIhE10hipError_tPPT_m.exit
.Ltmp105:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	leaq	80(%rsp), %rdi
	movl	$2048, %esi                     # imm = 0x800
	callq	hipMalloc@PLT
.Ltmp106:                               # EH_LABEL
# %bb.25:                               # %_ZL9hipMallocIhE10hipError_tPPT_m.exit124
	movq	104(%rsp), %rdi
.Ltmp107:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movq	%r15, %rsi
	movq	%r13, %rdx
	movl	$1, %ecx
	callq	hipMemcpy@PLT
.Ltmp108:                               # EH_LABEL
# %bb.26:
	movq	88(%rsp), %rdi
.Ltmp109:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movq	32(%rsp), %rsi                  # 8-byte Reload
	movq	296(%rsp), %rdx                 # 8-byte Reload
	movl	$1, %ecx
	callq	hipMemcpy@PLT
.Ltmp110:                               # EH_LABEL
# %bb.27:
	movq	80(%rsp), %rdi
.Ltmp111:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movl	$2048, %edx                     # imm = 0x800
	movq	72(%rsp), %rsi                  # 8-byte Reload
	movl	$1, %ecx
	callq	hipMemcpy@PLT
.Ltmp112:                               # EH_LABEL
# %bb.28:
	movq	8(%rsp), %rcx                   # 8-byte Reload
	leal	15(%rcx), %eax
	addl	$30, %ecx
	testl	%eax, %eax
	cmovnsl	%eax, %ecx
	sarl	$4, %ecx
	movq	48(%rsp), %rdx                  # 8-byte Reload
	leal	15(%rdx), %eax
	leal	30(%rdx), %ebx
	testl	%eax, %eax
	cmovnsl	%eax, %ebx
	sarl	$4, %ebx
	shlq	$32, %rbx
	orq	%rcx, %rbx
	movl	$20, %r14d
	movabsq	$4294967328, %r13               # imm = 0x100000020
	leaq	224(%rsp), %rbp
	.p2align	4
.LBB11_29:                              # =>This Inner Loop Header: Depth=1
	movq	96(%rsp), %rdi
.Ltmp114:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	xorl	%esi, %esi
	movq	%r12, %rdx
	callq	hipMemset@PLT
.Ltmp115:                               # EH_LABEL
# %bb.30:                               #   in Loop: Header=BB11_29 Depth=1
.Ltmp116:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movq	%rbx, %rdi
	movl	$1, %esi
	movq	%r13, %rdx
	movl	$1, %ecx
	xorl	%r8d, %r8d
	xorl	%r9d, %r9d
	callq	__hipPushCallConfiguration@PLT
.Ltmp117:                               # EH_LABEL
# %bb.31:                               #   in Loop: Header=BB11_29 Depth=1
	testl	%eax, %eax
	jne	.LBB11_34
# %bb.32:                               #   in Loop: Header=BB11_29 Depth=1
	movq	104(%rsp), %rax
	movq	%rax, 208(%rsp)
	movq	88(%rsp), %rax
	movq	%rax, 200(%rsp)
	movq	80(%rsp), %rax
	movq	%rax, 192(%rsp)
	movq	96(%rsp), %rax
	movq	%rax, 184(%rsp)
	movq	48(%rsp), %rax                  # 8-byte Reload
	movl	%eax, 56(%rsp)
	movq	8(%rsp), %rax                   # 8-byte Reload
	movl	%eax, 40(%rsp)
	movq	128(%rsp), %rax                 # 8-byte Reload
	movl	%eax, 28(%rsp)
	movl	68(%rsp), %eax                  # 4-byte Reload
	movl	%eax, 24(%rsp)
	leaq	208(%rsp), %rax
	movq	%rax, 224(%rsp)
	leaq	200(%rsp), %rax
	movq	%rax, 232(%rsp)
	leaq	192(%rsp), %rax
	movq	%rax, 240(%rsp)
	leaq	184(%rsp), %rax
	movq	%rax, 248(%rsp)
	leaq	56(%rsp), %rax
	movq	%rax, 256(%rsp)
	leaq	40(%rsp), %rax
	movq	%rax, 264(%rsp)
	leaq	28(%rsp), %rax
	movq	%rax, 272(%rsp)
	leaq	24(%rsp), %rax
	movq	%rax, 280(%rsp)
.Ltmp118:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	leaq	168(%rsp), %rdi
	leaq	152(%rsp), %rsi
	leaq	144(%rsp), %rdx
	leaq	136(%rsp), %rcx
	callq	__hipPopCallConfiguration@PLT
.Ltmp119:                               # EH_LABEL
# %bb.33:                               # %.noexc125
                                        #   in Loop: Header=BB11_29 Depth=1
	movq	168(%rsp), %rsi
	movl	176(%rsp), %edx
	movq	152(%rsp), %rcx
	movl	160(%rsp), %r8d
.Ltmp120:                               # EH_LABEL
	.cfi_escape 0x2e, 0x10
	movq	_Z12fused_m5_k32PK14__hip_fp8_e4m3PKhS3_Pfiiii@GOTPCREL(%rip), %rdi
	movq	%rbp, %r9
	pushq	136(%rsp)
	.cfi_adjust_cfa_offset 8
	pushq	152(%rsp)
	.cfi_adjust_cfa_offset 8
	callq	hipLaunchKernel@PLT
	addq	$16, %rsp
	.cfi_adjust_cfa_offset -16
.Ltmp121:                               # EH_LABEL
.LBB11_34:                              #   in Loop: Header=BB11_29 Depth=1
.Ltmp122:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	callq	hipDeviceSynchronize@PLT
.Ltmp123:                               # EH_LABEL
# %bb.35:                               #   in Loop: Header=BB11_29 Depth=1
	decl	%r14d
	jne	.LBB11_29
# %bb.36:
.Ltmp125:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	leaq	56(%rsp), %rdi
	callq	hipEventCreate@PLT
.Ltmp126:                               # EH_LABEL
# %bb.37:
.Ltmp127:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	leaq	40(%rsp), %rdi
	callq	hipEventCreate@PLT
.Ltmp128:                               # EH_LABEL
# %bb.38:
	movq	56(%rsp), %rdi
.Ltmp129:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	xorl	%esi, %esi
	callq	hipEventRecord@PLT
.Ltmp130:                               # EH_LABEL
# %bb.39:                               # %.preheader
	cmpl	$0, 20(%rsp)                    # 4-byte Folded Reload
	jle	.LBB11_46
# %bb.40:
	movabsq	$4294967328, %r14               # imm = 0x100000020
	movq	_Z12fused_m5_k32PK14__hip_fp8_e4m3PKhS3_Pfiiii@GOTPCREL(%rip), %r13
	leaq	224(%rsp), %rbp
	movl	20(%rsp), %r12d                 # 4-byte Reload
	jmp	.LBB11_41
	.p2align	4
.LBB11_45:                              #   in Loop: Header=BB11_41 Depth=1
	decl	%r12d
	je	.LBB11_46
.LBB11_41:                              # =>This Inner Loop Header: Depth=1
.Ltmp131:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movq	%rbx, %rdi
	movl	$1, %esi
	movq	%r14, %rdx
	movl	$1, %ecx
	xorl	%r8d, %r8d
	xorl	%r9d, %r9d
	callq	__hipPushCallConfiguration@PLT
.Ltmp132:                               # EH_LABEL
# %bb.42:                               #   in Loop: Header=BB11_41 Depth=1
	testl	%eax, %eax
	jne	.LBB11_45
# %bb.43:                               #   in Loop: Header=BB11_41 Depth=1
	movq	104(%rsp), %rax
	movq	%rax, 208(%rsp)
	movq	88(%rsp), %rax
	movq	%rax, 200(%rsp)
	movq	80(%rsp), %rax
	movq	%rax, 192(%rsp)
	movq	96(%rsp), %rax
	movq	%rax, 184(%rsp)
	movq	48(%rsp), %rax                  # 8-byte Reload
	movl	%eax, 28(%rsp)
	movq	8(%rsp), %rax                   # 8-byte Reload
	movl	%eax, 24(%rsp)
	movq	128(%rsp), %rax                 # 8-byte Reload
	movl	%eax, 220(%rsp)
	movl	68(%rsp), %eax                  # 4-byte Reload
	movl	%eax, 216(%rsp)
	leaq	208(%rsp), %rax
	movq	%rax, 224(%rsp)
	leaq	200(%rsp), %rax
	movq	%rax, 232(%rsp)
	leaq	192(%rsp), %rax
	movq	%rax, 240(%rsp)
	leaq	184(%rsp), %rax
	movq	%rax, 248(%rsp)
	leaq	28(%rsp), %rax
	movq	%rax, 256(%rsp)
	leaq	24(%rsp), %rax
	movq	%rax, 264(%rsp)
	leaq	220(%rsp), %rax
	movq	%rax, 272(%rsp)
	leaq	216(%rsp), %rax
	movq	%rax, 280(%rsp)
.Ltmp133:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	leaq	168(%rsp), %rdi
	leaq	152(%rsp), %rsi
	leaq	144(%rsp), %rdx
	leaq	136(%rsp), %rcx
	callq	__hipPopCallConfiguration@PLT
.Ltmp134:                               # EH_LABEL
# %bb.44:                               # %.noexc133
                                        #   in Loop: Header=BB11_41 Depth=1
	movq	168(%rsp), %rsi
	movl	176(%rsp), %edx
	movq	152(%rsp), %rcx
	movl	160(%rsp), %r8d
.Ltmp135:                               # EH_LABEL
	.cfi_escape 0x2e, 0x10
	movq	%r13, %rdi
	movq	%rbp, %r9
	pushq	136(%rsp)
	.cfi_adjust_cfa_offset 8
	pushq	152(%rsp)
	.cfi_adjust_cfa_offset 8
	callq	hipLaunchKernel@PLT
	addq	$16, %rsp
	.cfi_adjust_cfa_offset -16
.Ltmp136:                               # EH_LABEL
	jmp	.LBB11_45
.LBB11_46:                              # %._crit_edge203
	movq	40(%rsp), %rdi
.Ltmp138:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	xorl	%esi, %esi
	callq	hipEventRecord@PLT
.Ltmp139:                               # EH_LABEL
# %bb.47:
	movq	40(%rsp), %rdi
.Ltmp140:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	callq	hipEventSynchronize@PLT
.Ltmp141:                               # EH_LABEL
# %bb.48:
	movq	56(%rsp), %rsi
	movq	40(%rsp), %rdx
.Ltmp143:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	leaq	224(%rsp), %rdi
	callq	hipEventElapsedTime@PLT
.Ltmp144:                               # EH_LABEL
# %bb.49:
	cvtsi2ssl	20(%rsp), %xmm0         # 4-byte Folded Reload
	movss	224(%rsp), %xmm1                # xmm1 = mem[0],zero,zero,zero
	divss	%xmm0, %xmm1
	movss	%xmm1, 224(%rsp)
	movq	104(%rsp), %rdi
.Ltmp145:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	callq	hipFree@PLT
.Ltmp146:                               # EH_LABEL
# %bb.50:
	movq	96(%rsp), %rdi
.Ltmp147:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	callq	hipFree@PLT
.Ltmp148:                               # EH_LABEL
# %bb.51:
	movq	88(%rsp), %rdi
.Ltmp149:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	callq	hipFree@PLT
.Ltmp150:                               # EH_LABEL
# %bb.52:
	movq	80(%rsp), %rdi
.Ltmp151:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	callq	hipFree@PLT
.Ltmp152:                               # EH_LABEL
# %bb.53:
	movq	56(%rsp), %rdi
.Ltmp153:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	callq	hipEventDestroy@PLT
.Ltmp154:                               # EH_LABEL
# %bb.54:
	movq	40(%rsp), %rdi
.Ltmp155:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	callq	hipEventDestroy@PLT
.Ltmp156:                               # EH_LABEL
# %bb.55:
	movss	224(%rsp), %xmm0                # xmm0 = mem[0],zero,zero,zero
	movss	%xmm0, 8(%rsp)                  # 4-byte Spill
	testq	%r15, %r15
	je	.LBB11_57
# %bb.56:
	movq	112(%rsp), %rsi                 # 8-byte Reload
	subq	%r15, %rsi
	.cfi_escape 0x2e, 0x00
	movq	%r15, %rdi
	callq	_ZdlPvm@PLT
.LBB11_57:                              # %_ZNSt6vectorIhSaIhEED2Ev.exit
	.cfi_escape 0x2e, 0x00
	movl	$2048, %esi                     # imm = 0x800
	movq	72(%rsp), %rdi                  # 8-byte Reload
	callq	_ZdlPvm@PLT
	movq	32(%rsp), %rdi                  # 8-byte Reload
	testq	%rdi, %rdi
	je	.LBB11_59
# %bb.58:
	movq	120(%rsp), %rsi                 # 8-byte Reload
	subq	%rdi, %rsi
	.cfi_escape 0x2e, 0x00
	callq	_ZdlPvm@PLT
.LBB11_59:                              # %_ZNSt6vectorIhSaIhEED2Ev.exit138
	movss	8(%rsp), %xmm0                  # 4-byte Reload
                                        # xmm0 = mem[0],zero,zero,zero
	cvtss2sd	%xmm0, %xmm0
	addq	$5304, %rsp                     # imm = 0x14B8
	.cfi_def_cfa_offset 56
	popq	%rbx
	.cfi_def_cfa_offset 48
	popq	%r12
	.cfi_def_cfa_offset 40
	popq	%r13
	.cfi_def_cfa_offset 32
	popq	%r14
	.cfi_def_cfa_offset 24
	popq	%r15
	.cfi_def_cfa_offset 16
	popq	%rbp
	.cfi_def_cfa_offset 8
	retq
.LBB11_84:                              # %.noexc
	.cfi_def_cfa_offset 5360
	.cfi_escape 0x2e, 0x00
	leaq	.L.str.55(%rip), %rdi
	callq	_ZSt20__throw_length_errorPKc@PLT
.LBB11_13:
.Ltmp158:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	leaq	.L.str.55(%rip), %rdi
	callq	_ZSt20__throw_length_errorPKc@PLT
.Ltmp159:                               # EH_LABEL
# %bb.18:                               # %.noexc118
.LBB11_67:
.Ltmp87:                                # EH_LABEL
	movq	%rax, %rbx
	jmp	.LBB11_81
.LBB11_69:
.Ltmp160:                               # EH_LABEL
	movq	%rax, %rbx
	jmp	.LBB11_80
.LBB11_85:
.Ltmp142:                               # EH_LABEL
	jmp	.LBB11_78
.LBB11_77:
.Ltmp157:                               # EH_LABEL
	jmp	.LBB11_78
.LBB11_75:
.Ltmp113:                               # EH_LABEL
	jmp	.LBB11_78
.LBB11_73:
.Ltmp95:                                # EH_LABEL
	movq	%rax, %rbx
	jmp	.LBB11_79
.LBB11_74:
.Ltmp98:                                # EH_LABEL
	movq	%rax, %rbx
	jmp	.LBB11_79
.LBB11_17:                              # %_ZNSt6vectorIhSaIhEED2Ev.exit142.thread
.Ltmp84:                                # EH_LABEL
	movq	%rax, %rbx
	movq	32(%rsp), %rdi                  # 8-byte Reload
	jmp	.LBB11_82
.LBB11_68:
.Ltmp90:                                # EH_LABEL
	movq	%rax, %rbx
	jmp	.LBB11_80
.LBB11_86:
.Ltmp137:                               # EH_LABEL
	jmp	.LBB11_78
.LBB11_76:
.Ltmp124:                               # EH_LABEL
.LBB11_78:
	movq	%rax, %rbx
	testq	%r15, %r15
	je	.LBB11_80
.LBB11_79:                              # %.thread
	movq	112(%rsp), %rsi                 # 8-byte Reload
	subq	%r15, %rsi
	.cfi_escape 0x2e, 0x00
	movq	%r15, %rdi
	callq	_ZdlPvm@PLT
.LBB11_80:                              # %_ZNSt6vectorI14__hip_fp8_e4m3SaIS0_EED2Ev.exit140
	.cfi_escape 0x2e, 0x00
	movl	$2048, %esi                     # imm = 0x800
	movq	72(%rsp), %rdi                  # 8-byte Reload
	callq	_ZdlPvm@PLT
.LBB11_81:                              # %_ZNSt6vectorIhSaIhEED2Ev.exit142
	movq	32(%rsp), %rdi                  # 8-byte Reload
	testq	%rdi, %rdi
	je	.LBB11_83
.LBB11_82:
	movq	120(%rsp), %rsi                 # 8-byte Reload
	subq	%rdi, %rsi
	.cfi_escape 0x2e, 0x00
	callq	_ZdlPvm@PLT
.LBB11_83:                              # %_ZNSt6vectorIhSaIhEED2Ev.exit144
	.cfi_escape 0x2e, 0x00
	movq	%rbx, %rdi
	callq	_Unwind_Resume@PLT
.Lfunc_end11:
	.size	_Z14bench_fused_m5iiii, .Lfunc_end11-_Z14bench_fused_m5iiii
	.cfi_endproc
	.section	.gcc_except_table,"a",@progbits
	.p2align	2, 0x0
GCC_except_table11:
.Lexception1:
	.byte	255                             # @LPStart Encoding = omit
	.byte	255                             # @TType Encoding = omit
	.byte	1                               # Call site Encoding = uleb128
	.uleb128 .Lcst_end1-.Lcst_begin1
.Lcst_begin1:
	.uleb128 .Lfunc_begin1-.Lfunc_begin1    # >> Call Site 1 <<
	.uleb128 .Ltmp82-.Lfunc_begin1          #   Call between .Lfunc_begin1 and .Ltmp82
	.byte	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp82-.Lfunc_begin1          # >> Call Site 2 <<
	.uleb128 .Ltmp83-.Ltmp82                #   Call between .Ltmp82 and .Ltmp83
	.uleb128 .Ltmp84-.Lfunc_begin1          #     jumps to .Ltmp84
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp85-.Lfunc_begin1          # >> Call Site 3 <<
	.uleb128 .Ltmp86-.Ltmp85                #   Call between .Ltmp85 and .Ltmp86
	.uleb128 .Ltmp87-.Lfunc_begin1          #     jumps to .Ltmp87
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp86-.Lfunc_begin1          # >> Call Site 4 <<
	.uleb128 .Ltmp88-.Ltmp86                #   Call between .Ltmp86 and .Ltmp88
	.byte	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp88-.Lfunc_begin1          # >> Call Site 5 <<
	.uleb128 .Ltmp89-.Ltmp88                #   Call between .Ltmp88 and .Ltmp89
	.uleb128 .Ltmp90-.Lfunc_begin1          #     jumps to .Ltmp90
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp91-.Lfunc_begin1          # >> Call Site 6 <<
	.uleb128 .Ltmp92-.Ltmp91                #   Call between .Ltmp91 and .Ltmp92
	.uleb128 .Ltmp160-.Lfunc_begin1         #     jumps to .Ltmp160
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp92-.Lfunc_begin1          # >> Call Site 7 <<
	.uleb128 .Ltmp93-.Ltmp92                #   Call between .Ltmp92 and .Ltmp93
	.byte	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp93-.Lfunc_begin1          # >> Call Site 8 <<
	.uleb128 .Ltmp94-.Ltmp93                #   Call between .Ltmp93 and .Ltmp94
	.uleb128 .Ltmp95-.Lfunc_begin1          #     jumps to .Ltmp95
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp96-.Lfunc_begin1          # >> Call Site 9 <<
	.uleb128 .Ltmp97-.Ltmp96                #   Call between .Ltmp96 and .Ltmp97
	.uleb128 .Ltmp98-.Lfunc_begin1          #     jumps to .Ltmp98
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp99-.Lfunc_begin1          # >> Call Site 10 <<
	.uleb128 .Ltmp112-.Ltmp99               #   Call between .Ltmp99 and .Ltmp112
	.uleb128 .Ltmp113-.Lfunc_begin1         #     jumps to .Ltmp113
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp114-.Lfunc_begin1         # >> Call Site 11 <<
	.uleb128 .Ltmp123-.Ltmp114              #   Call between .Ltmp114 and .Ltmp123
	.uleb128 .Ltmp124-.Lfunc_begin1         #     jumps to .Ltmp124
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp125-.Lfunc_begin1         # >> Call Site 12 <<
	.uleb128 .Ltmp130-.Ltmp125              #   Call between .Ltmp125 and .Ltmp130
	.uleb128 .Ltmp142-.Lfunc_begin1         #     jumps to .Ltmp142
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp131-.Lfunc_begin1         # >> Call Site 13 <<
	.uleb128 .Ltmp136-.Ltmp131              #   Call between .Ltmp131 and .Ltmp136
	.uleb128 .Ltmp137-.Lfunc_begin1         #     jumps to .Ltmp137
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp138-.Lfunc_begin1         # >> Call Site 14 <<
	.uleb128 .Ltmp141-.Ltmp138              #   Call between .Ltmp138 and .Ltmp141
	.uleb128 .Ltmp142-.Lfunc_begin1         #     jumps to .Ltmp142
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp143-.Lfunc_begin1         # >> Call Site 15 <<
	.uleb128 .Ltmp156-.Ltmp143              #   Call between .Ltmp143 and .Ltmp156
	.uleb128 .Ltmp157-.Lfunc_begin1         #     jumps to .Ltmp157
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp156-.Lfunc_begin1         # >> Call Site 16 <<
	.uleb128 .Ltmp158-.Ltmp156              #   Call between .Ltmp156 and .Ltmp158
	.byte	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp158-.Lfunc_begin1         # >> Call Site 17 <<
	.uleb128 .Ltmp159-.Ltmp158              #   Call between .Ltmp158 and .Ltmp159
	.uleb128 .Ltmp160-.Lfunc_begin1         #     jumps to .Ltmp160
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp159-.Lfunc_begin1         # >> Call Site 18 <<
	.uleb128 .Lfunc_end11-.Ltmp159          #   Call between .Ltmp159 and .Lfunc_end11
	.byte	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
.Lcst_end1:
	.p2align	2, 0x0
                                        # -- End function
	.section	.rodata.cst4,"aM",@progbits,4
	.p2align	2, 0x0                          # -- Begin function _Z14bench_fused_m6iiiii
.LCPI12_0:
	.long	0x3f000000                      # float 0.5
.LCPI12_1:
	.long	0x41200000                      # float 10
	.text
	.globl	_Z14bench_fused_m6iiiii
	.prefalign	4, .Lfunc_end12, nop
	.type	_Z14bench_fused_m6iiiii,@function
_Z14bench_fused_m6iiiii:                # @_Z14bench_fused_m6iiiii
.Lfunc_begin2:
	.cfi_startproc
	.cfi_personality 155, DW.ref.__gxx_personality_v0
	.cfi_lsda 27, .Lexception2
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	pushq	%r15
	.cfi_def_cfa_offset 24
	pushq	%r14
	.cfi_def_cfa_offset 32
	pushq	%r13
	.cfi_def_cfa_offset 40
	pushq	%r12
	.cfi_def_cfa_offset 48
	pushq	%rbx
	.cfi_def_cfa_offset 56
	subq	$5432, %rsp                     # imm = 0x1538
	.cfi_def_cfa_offset 5488
	.cfi_offset %rbx, -56
	.cfi_offset %r12, -48
	.cfi_offset %r13, -40
	.cfi_offset %r14, -32
	.cfi_offset %r15, -24
	.cfi_offset %rbp, -16
	movl	%r8d, 200(%rsp)                 # 4-byte Spill
	movl	%ecx, %ebp
                                        # kill: def $edx killed $edx def $rdx
	movl	%esi, %r12d
                                        # kill: def $edi killed $edi def $rdi
	movq	%rdi, 128(%rsp)                 # 8-byte Spill
	leal	255(%rdx), %esi
	testl	%edx, %edx
	movq	%rdx, 16(%rsp)                  # 8-byte Spill
	cmovnsl	%edx, %esi
	sarl	$8, %esi
	movq	$2, 432(%rsp)
	movl	$2, %ecx
	movl	$2, %eax
	.p2align	4
.LBB12_1:                               # =>This Inner Loop Header: Depth=1
	movq	%rcx, %rdx
	shrq	$30, %rdx
	xorq	%rcx, %rdx
	imulq	$1812433253, %rdx, %rcx         # imm = 0x6C078965
	addq	%rax, %rcx
	decq	%rcx
	movl	%ecx, %edx
	movq	%rdx, 424(%rsp,%rax,8)
	cmpq	$624, %rax                      # imm = 0x270
	je	.LBB12_3
# %bb.2:                                #   in Loop: Header=BB12_1 Depth=1
	shrl	$30, %edx
	xorl	%edx, %ecx
	imull	$1812433253, %ecx, %ecx         # imm = 0x6C078965
	addl	%eax, %ecx
	movq	%rcx, 432(%rsp,%rax,8)
	addq	$2, %rax
	jmp	.LBB12_1
.LBB12_3:                               # %_ZNSt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EEC2Em.exit
	shll	$7, %esi
	movq	$624, 5424(%rsp)                # imm = 0x270
	movl	%esi, %eax
	imull	%r12d, %eax
	testl	%eax, %eax
	js	.LBB12_143
# %bb.4:                                # %_ZNSt6vectorIhSaIhEE17_S_check_init_lenEmRKS0_.exit.i
	movl	%esi, 152(%rsp)                 # 4-byte Spill
	je	.LBB12_5
# %bb.14:                               # %_ZNSt6vectorIhSaIhEEC2EmRKS0_.exit
	movslq	%eax, %r14
	.cfi_escape 0x2e, 0x00
	movq	%r14, %rdi
	callq	_Znwm@PLT
	movq	%rax, %r13
	addq	%r14, %rax
	movq	%rax, 328(%rsp)                 # 8-byte Spill
	movb	$0, (%r13)
	leaq	1(%r13), %rdi
	leaq	-1(%r14), %rdx
	.cfi_escape 0x2e, 0x00
	xorl	%ebx, %ebx
	xorl	%esi, %esi
	callq	memset@PLT
	leaq	432(%rsp), %r15
	.p2align	4
.LBB12_15:                              # %.lr.ph
                                        # =>This Inner Loop Header: Depth=1
.Ltmp161:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movq	%r15, %rdi
	callq	_ZNSt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EEclEv
.Ltmp162:                               # EH_LABEL
# %bb.16:                               #   in Loop: Header=BB12_15 Depth=1
	movb	%al, (%r13,%rbx)
	incq	%rbx
	cmpq	%rbx, %r14
	jne	.LBB12_15
	jmp	.LBB12_6
.LBB12_5:
	xorl	%r13d, %r13d
	movq	$0, 328(%rsp)                   # 8-byte Folded Spill
.LBB12_6:                               # %._crit_edge
.Ltmp164:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movl	$2048, %edi                     # imm = 0x800
	callq	_Znwm@PLT
.Ltmp165:                               # EH_LABEL
# %bb.7:                                # %.lr.ph377.preheader
	.cfi_escape 0x2e, 0x00
	xorl	%ebx, %ebx
	movl	$2048, %edx                     # imm = 0x800
	movq	%rax, 136(%rsp)                 # 8-byte Spill
	movq	%rax, %rdi
	xorl	%esi, %esi
	callq	memset@PLT
	leaq	432(%rsp), %r14
	movabsq	$-3689348814741910323, %r15     # imm = 0xCCCCCCCCCCCCCCCD
	jmp	.LBB12_8
	.p2align	4
.LBB12_11:                              #   in Loop: Header=BB12_8 Depth=1
	incq	%rbx
	cmpq	$2048, %rbx                     # imm = 0x800
	je	.LBB12_12
.LBB12_8:                               # %.lr.ph377
                                        # =>This Inner Loop Header: Depth=1
.Ltmp167:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movq	%r14, %rdi
	callq	_ZNSt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EEclEv
.Ltmp168:                               # EH_LABEL
# %bb.9:                                #   in Loop: Header=BB12_8 Depth=1
	movq	%rax, %rcx
	mulq	%r15
	shrq	$2, %rdx
	leal	(%rdx,%rdx,4), %eax
	subl	%eax, %ecx
	xorps	%xmm0, %xmm0
	cvtsi2ss	%ecx, %xmm0
	mulss	.LCPI12_0(%rip), %xmm0
	mulss	.LCPI12_1(%rip), %xmm0
	cvttss2si	%xmm0, %eax
	movq	136(%rsp), %rcx                 # 8-byte Reload
	movb	%al, (%rcx,%rbx)
	orl	$128, %eax
	movzbl	%al, %eax
	cmpl	$255, %eax
	jne	.LBB12_11
# %bb.10:                               #   in Loop: Header=BB12_8 Depth=1
	movb	$0, (%rcx,%rbx)
	jmp	.LBB12_11
.LBB12_12:                              # %._crit_edge378
	movq	16(%rsp), %rax                  # 8-byte Reload
                                        # kill: def $eax killed $eax killed $rax
	imull	128(%rsp), %eax                 # 4-byte Folded Reload
	testl	%eax, %eax
	js	.LBB12_13
# %bb.19:                               # %_ZNSt6vectorI14__hip_fp8_e4m3SaIS0_EE17_S_check_init_lenEmRKS1_.exit.i
	movslq	%eax, %r14
	movl	%ebp, 204(%rsp)                 # 4-byte Spill
	movq	%r12, 120(%rsp)                 # 8-byte Spill
	movq	%r14, 360(%rsp)                 # 8-byte Spill
	je	.LBB12_20
# %bb.24:
.Ltmp170:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movq	%r14, %rdi
	callq	_Znwm@PLT
.Ltmp171:                               # EH_LABEL
# %bb.25:                               # %.noexc178
	movq	%rax, %rbx
	movb	$0, (%rax)
	movq	%r14, %rdx
	decq	%rdx
	je	.LBB12_27
# %bb.26:                               # %_ZNSt6vectorI14__hip_fp8_e4m3SaIS0_EEC2EmRKS1_.exit
	leaq	1(%rbx), %rdi
	.cfi_escape 0x2e, 0x00
	xorl	%esi, %esi
	callq	memset@PLT
.LBB12_27:                              # %.lr.ph381.preheader
	movq	%rbx, %r12
	addq	%r14, %rbx
	movq	%rbx, 320(%rsp)                 # 8-byte Spill
	xorl	%ebx, %ebx
	leaq	432(%rsp), %r14
	movabsq	$-2049638230412172401, %r15     # imm = 0xE38E38E38E38E38F
	leaq	208(%rsp), %rbp
	.p2align	4
.LBB12_28:                              # %.lr.ph381
                                        # =>This Inner Loop Header: Depth=1
.Ltmp172:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movq	%r14, %rdi
	callq	_ZNSt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EEclEv
.Ltmp173:                               # EH_LABEL
# %bb.29:                               #   in Loop: Header=BB12_28 Depth=1
	movq	%rax, %rcx
	mulq	%r15
	shrq	$3, %rdx
	leaq	(%rdx,%rdx,8), %rax
	negq	%rax
	addq	%rcx, %rax
	addq	$-4, %rax
	testq	%rax, %rax
	js	.LBB12_30
# %bb.35:                               #   in Loop: Header=BB12_28 Depth=1
	xorps	%xmm0, %xmm0
	cvtsi2ss	%rax, %xmm0
	jmp	.LBB12_36
	.p2align	4
.LBB12_30:                              #   in Loop: Header=BB12_28 Depth=1
	movq	%rax, %rcx
	shrq	%rcx
	andl	$1, %eax
	orq	%rcx, %rax
	xorps	%xmm0, %xmm0
	cvtsi2ss	%rax, %xmm0
	addss	%xmm0, %xmm0
.LBB12_36:                              #   in Loop: Header=BB12_28 Depth=1
	mulss	.LCPI12_0(%rip), %xmm0
.Ltmp175:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movq	%rbp, %rdi
	callq	_ZN14__hip_fp8_e4m3C2Ef
.Ltmp176:                               # EH_LABEL
# %bb.37:                               #   in Loop: Header=BB12_28 Depth=1
	movzbl	208(%rsp), %eax
	movb	%al, (%r12,%rbx)
	incq	%rbx
	cmpq	%rbx, 360(%rsp)                 # 8-byte Folded Reload
	jne	.LBB12_28
# %bb.38:
	movq	%r12, %r15
	movl	204(%rsp), %ebp                 # 4-byte Reload
	movq	120(%rsp), %r12                 # 8-byte Reload
	jmp	.LBB12_21
.LBB12_20:
	xorl	%r15d, %r15d
	movq	$0, 320(%rsp)                   # 8-byte Folded Spill
.LBB12_21:                              # %._crit_edge382
	leal	15(%r12), %ecx
	testl	%r12d, %r12d
	movl	%r12d, %eax
	movl	%ecx, 428(%rsp)                 # 4-byte Spill
	cmovsl	%ecx, %eax
	sarl	$4, %eax
	cmpl	$2, %ebp
	movq	%r13, 304(%rsp)                 # 8-byte Spill
	movq	%r15, 288(%rsp)                 # 8-byte Spill
	jae	.LBB12_55
# %bb.22:
	movq	16(%rsp), %rcx                  # 8-byte Reload
	leal	31(%rcx), %ebx
	testl	%ecx, %ecx
	cmovnsl	%ecx, %ebx
	sarl	$5, %ebx
	imull	%ebx, %eax
	movl	%eax, %ecx
	shll	$8, %ecx
	testl	%eax, %eax
	movl	%ecx, 316(%rsp)                 # 4-byte Spill
	je	.LBB12_23
# %bb.42:
	js	.LBB12_43
# %bb.45:
.Ltmp183:                               # EH_LABEL
	movslq	%ecx, %r14
	.cfi_escape 0x2e, 0x00
	movq	%r14, %rdi
	callq	_Znwm@PLT
.Ltmp184:                               # EH_LABEL
# %bb.46:                               # %.noexc179
	movq	%rax, %rbp
	movb	$0, (%rax)
	leaq	-1(%r14), %rdx
	leaq	1(%rax), %rdi
	.cfi_escape 0x2e, 0x00
	xorl	%esi, %esi
	callq	memset@PLT
	movq	%rbp, %rax
	addq	%r14, %rax
	movq	%rax, 280(%rsp)                 # 8-byte Spill
	jmp	.LBB12_47
.LBB12_55:
	movq	16(%rsp), %rcx                  # 8-byte Reload
	leal	63(%rcx), %ebx
	testl	%ecx, %ecx
	cmovnsl	%ecx, %ebx
	sarl	$6, %ebx
	imull	%ebx, %eax
	movl	%eax, %ecx
	shll	$9, %ecx
	testl	%eax, %eax
	movl	%ecx, 316(%rsp)                 # 4-byte Spill
	je	.LBB12_56
# %bb.57:
	js	.LBB12_58
# %bb.60:
.Ltmp178:                               # EH_LABEL
	movslq	%ecx, %r14
	.cfi_escape 0x2e, 0x00
	movq	%r14, %rdi
	callq	_Znwm@PLT
.Ltmp179:                               # EH_LABEL
# %bb.61:                               # %.noexc181
	movq	%rax, %rbp
	movb	$0, (%rax)
	leaq	-1(%r14), %rdx
	leaq	1(%rax), %rdi
	.cfi_escape 0x2e, 0x00
	xorl	%esi, %esi
	callq	memset@PLT
	movq	%rbp, %rax
	addq	%r14, %rax
	movq	%rax, 280(%rsp)                 # 8-byte Spill
	jmp	.LBB12_62
.LBB12_23:
	movq	$0, 280(%rsp)                   # 8-byte Folded Spill
	xorl	%ebp, %ebp
.LBB12_47:                              # %_ZNSt6vectorIhSaIhEE6resizeEm.exit
	cmpl	$16, %r12d
	movq	%rbp, 272(%rsp)                 # 8-byte Spill
	jl	.LBB12_69
# %bb.48:                               # %.preheader.lr.ph.i
                                        # kill: def $r12d killed $r12d killed $r12 def $r12
	shrl	$4, %r12d
	movslq	152(%rsp), %rax                 # 4-byte Folded Reload
	movq	%rax, 344(%rsp)                 # 8-byte Spill
	movl	%ebx, %eax
	movq	%rax, 336(%rsp)                 # 8-byte Spill
	leaq	240(%rbp), %rax
	movq	%rax, 296(%rsp)                 # 8-byte Spill
	movq	%r12, 352(%rsp)                 # 8-byte Spill
	movq	%r12, %r14
	shlq	$8, %r14
	movq	$0, 192(%rsp)                   # 8-byte Folded Spill
	movq	304(%rsp), %rbp                 # 8-byte Reload
	jmp	.LBB12_49
	.p2align	4
.LBB12_52:                              # %._crit_edge.i
                                        #   in Loop: Header=BB12_49 Depth=1
	movq	192(%rsp), %rcx                 # 8-byte Reload
	incq	%rcx
	addq	$256, 296(%rsp)                 # 8-byte Folded Spill
                                        # imm = 0x100
	movq	%rcx, 192(%rsp)                 # 8-byte Spill
	cmpq	352(%rsp), %rcx                 # 8-byte Folded Reload
	je	.LBB12_68
.LBB12_49:                              # %.preheader.i
                                        # =>This Loop Header: Depth=1
                                        #     Child Loop BB12_51 Depth 2
	cmpl	$32, 16(%rsp)                   # 4-byte Folded Reload
	jl	.LBB12_52
# %bb.50:                               # %.lr.ph.i
                                        #   in Loop: Header=BB12_49 Depth=1
	movq	192(%rsp), %r9                  # 8-byte Reload
	shlq	$4, %r9
	movq	%r9, %rax
	movq	344(%rsp), %r11                 # 8-byte Reload
	imulq	%r11, %rax
	movq	%rax, 152(%rsp)                 # 8-byte Spill
	leaq	1(%r9), %rax
	imulq	%r11, %rax
	movq	%rax, 416(%rsp)                 # 8-byte Spill
	leaq	2(%r9), %rax
	imulq	%r11, %rax
	movq	%rax, 408(%rsp)                 # 8-byte Spill
	leaq	3(%r9), %rax
	imulq	%r11, %rax
	movq	%rax, 400(%rsp)                 # 8-byte Spill
	leaq	4(%r9), %rax
	imulq	%r11, %rax
	movq	%rax, 392(%rsp)                 # 8-byte Spill
	leaq	5(%r9), %rax
	imulq	%r11, %rax
	movq	%rax, 384(%rsp)                 # 8-byte Spill
	leaq	6(%r9), %rax
	imulq	%r11, %rax
	movq	%rax, 376(%rsp)                 # 8-byte Spill
	leaq	7(%r9), %rax
	imulq	%r11, %rax
	movq	%rax, 368(%rsp)                 # 8-byte Spill
	leaq	8(%r9), %rdx
	imulq	%r11, %rdx
	leaq	9(%r9), %rax
	imulq	%r11, %rax
	leaq	10(%r9), %rsi
	imulq	%r11, %rsi
	leaq	11(%r9), %r8
	imulq	%r11, %r8
	leaq	12(%r9), %rcx
	imulq	%r11, %rcx
	leaq	13(%r9), %rdi
	imulq	%r11, %rdi
	leaq	14(%r9), %r10
	imulq	%r11, %r10
	orq	$15, %r9
	imulq	%r11, %r9
	movq	296(%rsp), %r11                 # 8-byte Reload
	movq	336(%rsp), %rbx                 # 8-byte Reload
	xorl	%r12d, %r12d
	.p2align	4
.LBB12_51:                              #   Parent Loop BB12_49 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	movl	%r12d, %r15d
	andl	$112, %r15d
	addq	%rbp, %r15
	movq	%r14, %r13
	movl	%r12d, %r14d
	andl	$1073741696, %r14d              # imm = 0x3FFFFF80
	addq	%r15, %r14
	movq	152(%rsp), %r15                 # 8-byte Reload
	movups	(%r15,%r14), %xmm0
	movups	%xmm0, -240(%r11)
	movq	416(%rsp), %r15                 # 8-byte Reload
	movups	(%r15,%r14), %xmm0
	movups	%xmm0, -224(%r11)
	movq	408(%rsp), %r15                 # 8-byte Reload
	movups	(%r15,%r14), %xmm0
	movups	%xmm0, -208(%r11)
	movq	400(%rsp), %r15                 # 8-byte Reload
	movups	(%r15,%r14), %xmm0
	movups	%xmm0, -192(%r11)
	movq	392(%rsp), %r15                 # 8-byte Reload
	movups	(%r15,%r14), %xmm0
	movups	%xmm0, -176(%r11)
	movq	384(%rsp), %r15                 # 8-byte Reload
	movups	(%r15,%r14), %xmm0
	movups	%xmm0, -160(%r11)
	movq	376(%rsp), %r15                 # 8-byte Reload
	movups	(%r15,%r14), %xmm0
	movups	%xmm0, -144(%r11)
	movq	368(%rsp), %r15                 # 8-byte Reload
	movups	(%r15,%r14), %xmm0
	movups	%xmm0, -128(%r11)
	movups	(%rdx,%r14), %xmm0
	movups	%xmm0, -112(%r11)
	movups	(%rax,%r14), %xmm0
	movups	%xmm0, -96(%r11)
	movups	(%rsi,%r14), %xmm0
	movups	%xmm0, -80(%r11)
	movups	(%r8,%r14), %xmm0
	movups	%xmm0, -64(%r11)
	movups	(%rcx,%r14), %xmm0
	movups	%xmm0, -48(%r11)
	movups	(%rdi,%r14), %xmm0
	movups	%xmm0, -32(%r11)
	movups	(%r10,%r14), %xmm0
	movups	%xmm0, -16(%r11)
	movups	(%r9,%r14), %xmm0
	movq	%r13, %r14
	movups	%xmm0, (%r11)
	addq	$16, %r12
	addq	%r13, %r11
	decq	%rbx
	jne	.LBB12_51
	jmp	.LBB12_52
.LBB12_56:
	movq	$0, 280(%rsp)                   # 8-byte Folded Spill
	xorl	%ebp, %ebp
.LBB12_62:                              # %_ZNSt6vectorIhSaIhEE6resizeEm.exit182
	cmpl	$16, %r12d
	movq	%rbp, 272(%rsp)                 # 8-byte Spill
	jl	.LBB12_69
# %bb.63:                               # %.preheader.lr.ph.i183
                                        # kill: def $r12d killed $r12d killed $r12 def $r12
	shrl	$4, %r12d
	movslq	152(%rsp), %rax                 # 4-byte Folded Reload
	movq	%rax, 344(%rsp)                 # 8-byte Spill
	movl	%ebx, %eax
	movq	%rax, 336(%rsp)                 # 8-byte Spill
	leaq	496(%rbp), %rax
	movq	%rax, 296(%rsp)                 # 8-byte Spill
	movq	%r12, 352(%rsp)                 # 8-byte Spill
	movq	%r12, %r15
	shlq	$9, %r15
	movq	$0, 192(%rsp)                   # 8-byte Folded Spill
	jmp	.LBB12_64
	.p2align	4
.LBB12_67:                              # %._crit_edge.i188
                                        #   in Loop: Header=BB12_64 Depth=1
	movq	192(%rsp), %rcx                 # 8-byte Reload
	incq	%rcx
	addq	$512, 296(%rsp)                 # 8-byte Folded Spill
                                        # imm = 0x200
	movq	%rcx, 192(%rsp)                 # 8-byte Spill
	cmpq	352(%rsp), %rcx                 # 8-byte Folded Reload
	je	.LBB12_68
.LBB12_64:                              # %.preheader.i186
                                        # =>This Loop Header: Depth=1
                                        #     Child Loop BB12_66 Depth 2
	cmpl	$64, 16(%rsp)                   # 4-byte Folded Reload
	movq	304(%rsp), %r12                 # 8-byte Reload
	jl	.LBB12_67
# %bb.65:                               # %.lr.ph.i191
                                        #   in Loop: Header=BB12_64 Depth=1
	movq	192(%rsp), %r9                  # 8-byte Reload
	shlq	$4, %r9
	movq	%r9, %rax
	movq	344(%rsp), %rdx                 # 8-byte Reload
	imulq	%rdx, %rax
	movq	%rax, 152(%rsp)                 # 8-byte Spill
	leaq	1(%r9), %rax
	imulq	%rdx, %rax
	movq	%rax, 416(%rsp)                 # 8-byte Spill
	leaq	2(%r9), %rax
	imulq	%rdx, %rax
	movq	%rax, 408(%rsp)                 # 8-byte Spill
	leaq	3(%r9), %rax
	imulq	%rdx, %rax
	movq	%rax, 400(%rsp)                 # 8-byte Spill
	leaq	4(%r9), %rax
	imulq	%rdx, %rax
	movq	%rax, 392(%rsp)                 # 8-byte Spill
	leaq	5(%r9), %rax
	imulq	%rdx, %rax
	movq	%rax, 384(%rsp)                 # 8-byte Spill
	leaq	6(%r9), %rax
	imulq	%rdx, %rax
	movq	%rax, 376(%rsp)                 # 8-byte Spill
	leaq	7(%r9), %rax
	imulq	%rdx, %rax
	movq	%rax, 368(%rsp)                 # 8-byte Spill
	leaq	8(%r9), %r13
	imulq	%rdx, %r13
	leaq	9(%r9), %rbx
	imulq	%rdx, %rbx
	leaq	10(%r9), %rsi
	imulq	%rdx, %rsi
	leaq	11(%r9), %r8
	imulq	%rdx, %r8
	leaq	12(%r9), %rcx
	imulq	%rdx, %rcx
	leaq	13(%r9), %rdi
	imulq	%rdx, %rdi
	leaq	14(%r9), %r10
	imulq	%rdx, %r10
	orq	$15, %r9
	imulq	%rdx, %r9
	movq	296(%rsp), %r11                 # 8-byte Reload
	movq	336(%rsp), %r14                 # 8-byte Reload
	xorl	%edx, %edx
	.p2align	4
.LBB12_66:                              #   Parent Loop BB12_64 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	movl	%edx, %ebp
	andl	$96, %ebp
	addq	%r12, %rbp
	movq	%r12, %rax
	movq	%r15, %r12
	movl	%edx, %r15d
	andl	$1073741696, %r15d              # imm = 0x3FFFFF80
	addq	%rbp, %r15
	movq	152(%rsp), %rbp                 # 8-byte Reload
	movups	(%rbp,%r15), %xmm0
	movups	%xmm0, -496(%r11)
	movups	16(%rbp,%r15), %xmm0
	movups	%xmm0, -480(%r11)
	movq	416(%rsp), %rbp                 # 8-byte Reload
	movups	(%rbp,%r15), %xmm0
	movups	%xmm0, -464(%r11)
	movups	16(%rbp,%r15), %xmm0
	movups	%xmm0, -448(%r11)
	movq	408(%rsp), %rbp                 # 8-byte Reload
	movups	(%rbp,%r15), %xmm0
	movups	%xmm0, -432(%r11)
	movups	16(%rbp,%r15), %xmm0
	movups	%xmm0, -416(%r11)
	movq	400(%rsp), %rbp                 # 8-byte Reload
	movups	(%rbp,%r15), %xmm0
	movups	%xmm0, -400(%r11)
	movups	16(%rbp,%r15), %xmm0
	movups	%xmm0, -384(%r11)
	movq	392(%rsp), %rbp                 # 8-byte Reload
	movups	(%rbp,%r15), %xmm0
	movups	%xmm0, -368(%r11)
	movups	16(%rbp,%r15), %xmm0
	movups	%xmm0, -352(%r11)
	movq	384(%rsp), %rbp                 # 8-byte Reload
	movups	(%rbp,%r15), %xmm0
	movups	%xmm0, -336(%r11)
	movups	16(%rbp,%r15), %xmm0
	movups	%xmm0, -320(%r11)
	movq	376(%rsp), %rbp                 # 8-byte Reload
	movups	(%rbp,%r15), %xmm0
	movups	%xmm0, -304(%r11)
	movups	16(%rbp,%r15), %xmm0
	movups	%xmm0, -288(%r11)
	movq	368(%rsp), %rbp                 # 8-byte Reload
	movups	(%rbp,%r15), %xmm0
	movups	%xmm0, -272(%r11)
	movups	16(%rbp,%r15), %xmm0
	movups	%xmm0, -256(%r11)
	movups	(%r13,%r15), %xmm0
	movups	%xmm0, -240(%r11)
	movups	16(%r13,%r15), %xmm0
	movups	%xmm0, -224(%r11)
	movups	(%rbx,%r15), %xmm0
	movups	%xmm0, -208(%r11)
	movups	16(%rbx,%r15), %xmm0
	movups	%xmm0, -192(%r11)
	movups	(%rsi,%r15), %xmm0
	movups	%xmm0, -176(%r11)
	movups	16(%rsi,%r15), %xmm0
	movups	%xmm0, -160(%r11)
	movups	(%r8,%r15), %xmm0
	movups	%xmm0, -144(%r11)
	movups	16(%r8,%r15), %xmm0
	movups	%xmm0, -128(%r11)
	movups	(%rcx,%r15), %xmm0
	movups	%xmm0, -112(%r11)
	movups	16(%rcx,%r15), %xmm0
	movups	%xmm0, -96(%r11)
	movups	(%rdi,%r15), %xmm0
	movups	%xmm0, -80(%r11)
	movups	16(%rdi,%r15), %xmm0
	movups	%xmm0, -64(%r11)
	movups	(%r10,%r15), %xmm0
	movups	%xmm0, -48(%r11)
	movups	16(%r10,%r15), %xmm0
	movups	%xmm0, -32(%r11)
	movups	(%r9,%r15), %xmm0
	movups	%xmm0, -16(%r11)
	movups	16(%r9,%r15), %xmm0
	movq	%r12, %r15
	movq	%rax, %r12
	movups	%xmm0, (%r11)
	addq	$32, %rdx
	addq	%r15, %r11
	decq	%r14
	jne	.LBB12_66
	jmp	.LBB12_67
.LBB12_68:
	movq	304(%rsp), %r13                 # 8-byte Reload
	movq	288(%rsp), %r15                 # 8-byte Reload
	movq	120(%rsp), %r12                 # 8-byte Reload
.LBB12_69:                              # %_Z17repack_k32_for_m6PKhPhiii.exit
.Ltmp188:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	leaq	184(%rsp), %rdi
	movq	360(%rsp), %rsi                 # 8-byte Reload
	movq	136(%rsp), %rbx                 # 8-byte Reload
	callq	hipMalloc@PLT
.Ltmp189:                               # EH_LABEL
# %bb.70:                               # %_ZL9hipMallocI14__hip_fp8_e4m3E10hipError_tPPT_m.exit
	movl	%r12d, %eax
	imull	128(%rsp), %eax                 # 4-byte Folded Reload
	movslq	%eax, %r14
	shlq	$2, %r14
.Ltmp190:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	leaq	176(%rsp), %rdi
	movq	%r14, %rsi
	callq	hipMalloc@PLT
.Ltmp191:                               # EH_LABEL
# %bb.71:                               # %_ZL9hipMallocIfE10hipError_tPPT_m.exit
	movslq	316(%rsp), %rbp                 # 4-byte Folded Reload
.Ltmp192:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	leaq	168(%rsp), %rdi
	movq	%rbp, %rsi
	callq	hipMalloc@PLT
.Ltmp193:                               # EH_LABEL
# %bb.72:                               # %_ZL9hipMallocIhE10hipError_tPPT_m.exit
.Ltmp194:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	leaq	160(%rsp), %rdi
	movl	$2048, %esi                     # imm = 0x800
	callq	hipMalloc@PLT
.Ltmp195:                               # EH_LABEL
# %bb.73:                               # %_ZL9hipMallocIhE10hipError_tPPT_m.exit210
	movq	184(%rsp), %rdi
.Ltmp196:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movq	%r15, %rsi
	movq	360(%rsp), %rdx                 # 8-byte Reload
	movl	$1, %ecx
	callq	hipMemcpy@PLT
.Ltmp197:                               # EH_LABEL
# %bb.74:
	movq	168(%rsp), %rdi
.Ltmp198:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movq	272(%rsp), %rsi                 # 8-byte Reload
	movq	%rbp, %rdx
	movl	$1, %ecx
	callq	hipMemcpy@PLT
.Ltmp199:                               # EH_LABEL
# %bb.75:
	movq	160(%rsp), %rdi
.Ltmp200:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movl	$2048, %edx                     # imm = 0x800
	movq	%rbx, %rsi
	movl	$1, %ecx
	callq	hipMemcpy@PLT
.Ltmp201:                               # EH_LABEL
# %bb.76:
	leal	30(%r12), %eax
	movl	428(%rsp), %ecx                 # 4-byte Reload
	testl	%ecx, %ecx
	cmovnsl	%ecx, %eax
	sarl	$4, %eax
	movq	128(%rsp), %rdx                 # 8-byte Reload
	leal	15(%rdx), %ecx
	leal	30(%rdx), %r12d
	testl	%ecx, %ecx
	cmovnsl	%ecx, %r12d
	sarl	$4, %r12d
	shlq	$32, %r12
	orq	%rax, %r12
	movl	$20, %r15d
	movabsq	$4294967328, %rbp               # imm = 0x100000020
	.p2align	4
.LBB12_77:                              # =>This Inner Loop Header: Depth=1
	movq	176(%rsp), %rdi
.Ltmp203:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	xorl	%esi, %esi
	movq	%r14, %rdx
	callq	hipMemset@PLT
.Ltmp204:                               # EH_LABEL
# %bb.78:                               #   in Loop: Header=BB12_77 Depth=1
	movl	204(%rsp), %eax                 # 4-byte Reload
	testl	%eax, %eax
	je	.LBB12_79
# %bb.84:                               #   in Loop: Header=BB12_77 Depth=1
	cmpl	$1, %eax
	jne	.LBB12_89
# %bb.85:                               #   in Loop: Header=BB12_77 Depth=1
.Ltmp211:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movq	%r12, %rdi
	movl	$1, %esi
	movq	%rbp, %rdx
	movl	$1, %ecx
	xorl	%r8d, %r8d
	xorl	%r9d, %r9d
	callq	__hipPushCallConfiguration@PLT
.Ltmp212:                               # EH_LABEL
# %bb.86:                               #   in Loop: Header=BB12_77 Depth=1
	testl	%eax, %eax
	jne	.LBB12_93
# %bb.87:                               #   in Loop: Header=BB12_77 Depth=1
	movq	184(%rsp), %rax
	movq	168(%rsp), %rcx
	movq	160(%rsp), %rdx
	movq	176(%rsp), %rsi
	movq	%rax, 96(%rsp)
	movq	%rcx, 88(%rsp)
	movq	%rdx, 80(%rsp)
	movq	%rsi, 72(%rsp)
	movq	128(%rsp), %rax                 # 8-byte Reload
	movl	%eax, 144(%rsp)
	movq	120(%rsp), %rax                 # 8-byte Reload
	movl	%eax, 112(%rsp)
	movq	16(%rsp), %rax                  # 8-byte Reload
	movl	%eax, 12(%rsp)
	leaq	96(%rsp), %rax
	movq	%rax, 208(%rsp)
	leaq	88(%rsp), %rax
	movq	%rax, 216(%rsp)
	leaq	80(%rsp), %rax
	movq	%rax, 224(%rsp)
	leaq	72(%rsp), %rax
	movq	%rax, 232(%rsp)
	leaq	144(%rsp), %rax
	movq	%rax, 240(%rsp)
	leaq	112(%rsp), %rax
	movq	%rax, 248(%rsp)
	leaq	12(%rsp), %rax
	movq	%rax, 256(%rsp)
.Ltmp213:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	leaq	56(%rsp), %rdi
	leaq	40(%rsp), %rsi
	leaq	32(%rsp), %rdx
	leaq	24(%rsp), %rcx
	callq	__hipPopCallConfiguration@PLT
.Ltmp214:                               # EH_LABEL
# %bb.88:                               # %.noexc219
                                        #   in Loop: Header=BB12_77 Depth=1
	movq	56(%rsp), %rsi
	movl	64(%rsp), %edx
	movq	40(%rsp), %rcx
	movl	48(%rsp), %r8d
.Ltmp215:                               # EH_LABEL
	.cfi_escape 0x2e, 0x10
	movq	_Z24fused_m6_repacked_db_k32PK14__hip_fp8_e4m3PKhS3_Pfiii@GOTPCREL(%rip), %rdi
	leaq	208(%rsp), %r9
	pushq	24(%rsp)
	.cfi_adjust_cfa_offset 8
	pushq	40(%rsp)
	.cfi_adjust_cfa_offset 8
	callq	hipLaunchKernel@PLT
	addq	$16, %rsp
	.cfi_adjust_cfa_offset -16
.Ltmp216:                               # EH_LABEL
	jmp	.LBB12_93
	.p2align	4
.LBB12_79:                              #   in Loop: Header=BB12_77 Depth=1
.Ltmp217:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movq	%r12, %rdi
	movl	$1, %esi
	movq	%rbp, %rdx
	movl	$1, %ecx
	xorl	%r8d, %r8d
	xorl	%r9d, %r9d
	callq	__hipPushCallConfiguration@PLT
.Ltmp218:                               # EH_LABEL
# %bb.80:                               #   in Loop: Header=BB12_77 Depth=1
	testl	%eax, %eax
	jne	.LBB12_93
# %bb.81:                               #   in Loop: Header=BB12_77 Depth=1
	movq	184(%rsp), %rax
	movq	168(%rsp), %rcx
	movq	160(%rsp), %rdx
	movq	176(%rsp), %rsi
	movq	%rax, 96(%rsp)
	movq	%rcx, 88(%rsp)
	movq	%rdx, 80(%rsp)
	movq	%rsi, 72(%rsp)
	movq	128(%rsp), %rax                 # 8-byte Reload
	movl	%eax, 144(%rsp)
	movq	120(%rsp), %rax                 # 8-byte Reload
	movl	%eax, 112(%rsp)
	movq	16(%rsp), %rax                  # 8-byte Reload
	movl	%eax, 12(%rsp)
	leaq	96(%rsp), %rax
	movq	%rax, 208(%rsp)
	leaq	88(%rsp), %rax
	movq	%rax, 216(%rsp)
	leaq	80(%rsp), %rax
	movq	%rax, 224(%rsp)
	leaq	72(%rsp), %rax
	movq	%rax, 232(%rsp)
	leaq	144(%rsp), %rax
	movq	%rax, 240(%rsp)
	leaq	112(%rsp), %rax
	movq	%rax, 248(%rsp)
	leaq	12(%rsp), %rax
	movq	%rax, 256(%rsp)
.Ltmp219:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	leaq	56(%rsp), %rdi
	leaq	40(%rsp), %rsi
	leaq	32(%rsp), %rdx
	leaq	24(%rsp), %rcx
	callq	__hipPopCallConfiguration@PLT
.Ltmp220:                               # EH_LABEL
# %bb.82:                               # %.noexc211
                                        #   in Loop: Header=BB12_77 Depth=1
	movq	56(%rsp), %rsi
	movl	64(%rsp), %edx
	movq	40(%rsp), %rcx
	movl	48(%rsp), %r8d
.Ltmp221:                               # EH_LABEL
	.cfi_escape 0x2e, 0x10
	movq	_Z21fused_m6_repacked_k32PK14__hip_fp8_e4m3PKhS3_Pfiii@GOTPCREL(%rip), %rdi
	leaq	208(%rsp), %r9
	pushq	24(%rsp)
	.cfi_adjust_cfa_offset 8
	pushq	40(%rsp)
	.cfi_adjust_cfa_offset 8
	callq	hipLaunchKernel@PLT
	addq	$16, %rsp
	.cfi_adjust_cfa_offset -16
.Ltmp222:                               # EH_LABEL
	jmp	.LBB12_93
	.p2align	4
.LBB12_89:                              #   in Loop: Header=BB12_77 Depth=1
.Ltmp205:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movq	%r12, %rdi
	movl	$1, %esi
	movq	%rbp, %rdx
	movl	$1, %ecx
	xorl	%r8d, %r8d
	xorl	%r9d, %r9d
	callq	__hipPushCallConfiguration@PLT
.Ltmp206:                               # EH_LABEL
# %bb.90:                               #   in Loop: Header=BB12_77 Depth=1
	testl	%eax, %eax
	jne	.LBB12_93
# %bb.91:                               #   in Loop: Header=BB12_77 Depth=1
	movq	184(%rsp), %rax
	movq	168(%rsp), %rcx
	movq	160(%rsp), %rdx
	movq	176(%rsp), %rsi
	movq	%rax, 96(%rsp)
	movq	%rcx, 88(%rsp)
	movq	%rdx, 80(%rsp)
	movq	%rsi, 72(%rsp)
	movq	128(%rsp), %rax                 # 8-byte Reload
	movl	%eax, 144(%rsp)
	movq	120(%rsp), %rax                 # 8-byte Reload
	movl	%eax, 112(%rsp)
	movq	16(%rsp), %rax                  # 8-byte Reload
	movl	%eax, 12(%rsp)
	leaq	96(%rsp), %rax
	movq	%rax, 208(%rsp)
	leaq	88(%rsp), %rax
	movq	%rax, 216(%rsp)
	leaq	80(%rsp), %rax
	movq	%rax, 224(%rsp)
	leaq	72(%rsp), %rax
	movq	%rax, 232(%rsp)
	leaq	144(%rsp), %rax
	movq	%rax, 240(%rsp)
	leaq	112(%rsp), %rax
	movq	%rax, 248(%rsp)
	leaq	12(%rsp), %rax
	movq	%rax, 256(%rsp)
.Ltmp207:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	leaq	56(%rsp), %rdi
	leaq	40(%rsp), %rsi
	leaq	32(%rsp), %rdx
	leaq	24(%rsp), %rcx
	callq	__hipPopCallConfiguration@PLT
.Ltmp208:                               # EH_LABEL
# %bb.92:                               # %.noexc227
                                        #   in Loop: Header=BB12_77 Depth=1
	movq	56(%rsp), %rsi
	movl	64(%rsp), %edx
	movq	40(%rsp), %rcx
	movl	48(%rsp), %r8d
.Ltmp209:                               # EH_LABEL
	.cfi_escape 0x2e, 0x10
	movq	_Z21fused_m6_repacked_k64PK14__hip_fp8_e4m3PKhS3_Pfiii@GOTPCREL(%rip), %rdi
	leaq	208(%rsp), %r9
	pushq	24(%rsp)
	.cfi_adjust_cfa_offset 8
	pushq	40(%rsp)
	.cfi_adjust_cfa_offset 8
	callq	hipLaunchKernel@PLT
	addq	$16, %rsp
	.cfi_adjust_cfa_offset -16
.Ltmp210:                               # EH_LABEL
	.p2align	4
.LBB12_93:                              #   in Loop: Header=BB12_77 Depth=1
.Ltmp223:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	callq	hipDeviceSynchronize@PLT
.Ltmp224:                               # EH_LABEL
# %bb.94:                               #   in Loop: Header=BB12_77 Depth=1
	decl	%r15d
	jne	.LBB12_77
# %bb.95:
.Ltmp226:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	leaq	144(%rsp), %rdi
	callq	hipEventCreate@PLT
.Ltmp227:                               # EH_LABEL
# %bb.96:
.Ltmp228:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	leaq	112(%rsp), %rdi
	callq	hipEventCreate@PLT
.Ltmp229:                               # EH_LABEL
# %bb.97:
	movq	144(%rsp), %rdi
.Ltmp230:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	xorl	%esi, %esi
	callq	hipEventRecord@PLT
.Ltmp231:                               # EH_LABEL
# %bb.98:                               # %.preheader
	cmpl	$0, 200(%rsp)                   # 4-byte Folded Reload
	jle	.LBB12_116
# %bb.99:
	movabsq	$4294967328, %r14               # imm = 0x100000020
	leaq	208(%rsp), %rbp
	movl	200(%rsp), %r15d                # 4-byte Reload
	jmp	.LBB12_100
	.p2align	4
.LBB12_115:                             #   in Loop: Header=BB12_100 Depth=1
	decl	%r15d
	je	.LBB12_116
.LBB12_100:                             # =>This Inner Loop Header: Depth=1
	movl	204(%rsp), %eax                 # 4-byte Reload
	testl	%eax, %eax
	je	.LBB12_101
# %bb.106:                              #   in Loop: Header=BB12_100 Depth=1
	cmpl	$1, %eax
	jne	.LBB12_111
# %bb.107:                              #   in Loop: Header=BB12_100 Depth=1
.Ltmp238:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movq	%r12, %rdi
	movl	$1, %esi
	movq	%r14, %rdx
	movl	$1, %ecx
	xorl	%r8d, %r8d
	xorl	%r9d, %r9d
	callq	__hipPushCallConfiguration@PLT
.Ltmp239:                               # EH_LABEL
# %bb.108:                              #   in Loop: Header=BB12_100 Depth=1
	testl	%eax, %eax
	jne	.LBB12_115
# %bb.109:                              #   in Loop: Header=BB12_100 Depth=1
	movq	184(%rsp), %rax
	movq	168(%rsp), %rcx
	movq	160(%rsp), %rdx
	movq	176(%rsp), %rsi
	movq	%rax, 96(%rsp)
	movq	%rcx, 88(%rsp)
	movq	%rdx, 80(%rsp)
	movq	%rsi, 72(%rsp)
	movq	128(%rsp), %rax                 # 8-byte Reload
	movl	%eax, 12(%rsp)
	movq	120(%rsp), %rax                 # 8-byte Reload
	movl	%eax, 108(%rsp)
	movq	16(%rsp), %rax                  # 8-byte Reload
	movl	%eax, 104(%rsp)
	leaq	96(%rsp), %rax
	movq	%rax, 208(%rsp)
	leaq	88(%rsp), %rax
	movq	%rax, 216(%rsp)
	leaq	80(%rsp), %rax
	movq	%rax, 224(%rsp)
	leaq	72(%rsp), %rax
	movq	%rax, 232(%rsp)
	leaq	12(%rsp), %rax
	movq	%rax, 240(%rsp)
	leaq	108(%rsp), %rax
	movq	%rax, 248(%rsp)
	leaq	104(%rsp), %rax
	movq	%rax, 256(%rsp)
.Ltmp240:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	leaq	56(%rsp), %rdi
	leaq	40(%rsp), %rsi
	leaq	32(%rsp), %rdx
	leaq	24(%rsp), %rcx
	callq	__hipPopCallConfiguration@PLT
.Ltmp241:                               # EH_LABEL
# %bb.110:                              # %.noexc244
                                        #   in Loop: Header=BB12_100 Depth=1
	movq	56(%rsp), %rsi
	movl	64(%rsp), %edx
	movq	40(%rsp), %rcx
	movl	48(%rsp), %r8d
.Ltmp242:                               # EH_LABEL
	.cfi_escape 0x2e, 0x10
	movq	_Z24fused_m6_repacked_db_k32PK14__hip_fp8_e4m3PKhS3_Pfiii@GOTPCREL(%rip), %rdi
	movq	%rbp, %r9
	pushq	24(%rsp)
	.cfi_adjust_cfa_offset 8
	pushq	40(%rsp)
	.cfi_adjust_cfa_offset 8
	callq	hipLaunchKernel@PLT
	addq	$16, %rsp
	.cfi_adjust_cfa_offset -16
.Ltmp243:                               # EH_LABEL
	jmp	.LBB12_115
	.p2align	4
.LBB12_101:                             #   in Loop: Header=BB12_100 Depth=1
.Ltmp244:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movq	%r12, %rdi
	movl	$1, %esi
	movq	%r14, %rdx
	movl	$1, %ecx
	xorl	%r8d, %r8d
	xorl	%r9d, %r9d
	callq	__hipPushCallConfiguration@PLT
.Ltmp245:                               # EH_LABEL
# %bb.102:                              #   in Loop: Header=BB12_100 Depth=1
	testl	%eax, %eax
	jne	.LBB12_115
# %bb.103:                              #   in Loop: Header=BB12_100 Depth=1
	movq	184(%rsp), %rax
	movq	168(%rsp), %rcx
	movq	160(%rsp), %rdx
	movq	176(%rsp), %rsi
	movq	%rax, 96(%rsp)
	movq	%rcx, 88(%rsp)
	movq	%rdx, 80(%rsp)
	movq	%rsi, 72(%rsp)
	movq	128(%rsp), %rax                 # 8-byte Reload
	movl	%eax, 12(%rsp)
	movq	120(%rsp), %rax                 # 8-byte Reload
	movl	%eax, 108(%rsp)
	movq	16(%rsp), %rax                  # 8-byte Reload
	movl	%eax, 104(%rsp)
	leaq	96(%rsp), %rax
	movq	%rax, 208(%rsp)
	leaq	88(%rsp), %rax
	movq	%rax, 216(%rsp)
	leaq	80(%rsp), %rax
	movq	%rax, 224(%rsp)
	leaq	72(%rsp), %rax
	movq	%rax, 232(%rsp)
	leaq	12(%rsp), %rax
	movq	%rax, 240(%rsp)
	leaq	108(%rsp), %rax
	movq	%rax, 248(%rsp)
	leaq	104(%rsp), %rax
	movq	%rax, 256(%rsp)
.Ltmp246:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	leaq	56(%rsp), %rdi
	leaq	40(%rsp), %rsi
	leaq	32(%rsp), %rdx
	leaq	24(%rsp), %rcx
	callq	__hipPopCallConfiguration@PLT
.Ltmp247:                               # EH_LABEL
# %bb.104:                              # %.noexc235
                                        #   in Loop: Header=BB12_100 Depth=1
	movq	56(%rsp), %rsi
	movl	64(%rsp), %edx
	movq	40(%rsp), %rcx
	movl	48(%rsp), %r8d
.Ltmp248:                               # EH_LABEL
	.cfi_escape 0x2e, 0x10
	movq	_Z21fused_m6_repacked_k32PK14__hip_fp8_e4m3PKhS3_Pfiii@GOTPCREL(%rip), %rdi
	movq	%rbp, %r9
	pushq	24(%rsp)
	.cfi_adjust_cfa_offset 8
	pushq	40(%rsp)
	.cfi_adjust_cfa_offset 8
	callq	hipLaunchKernel@PLT
	addq	$16, %rsp
	.cfi_adjust_cfa_offset -16
.Ltmp249:                               # EH_LABEL
	jmp	.LBB12_115
	.p2align	4
.LBB12_111:                             #   in Loop: Header=BB12_100 Depth=1
.Ltmp232:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movq	%r12, %rdi
	movl	$1, %esi
	movq	%r14, %rdx
	movl	$1, %ecx
	xorl	%r8d, %r8d
	xorl	%r9d, %r9d
	callq	__hipPushCallConfiguration@PLT
.Ltmp233:                               # EH_LABEL
# %bb.112:                              #   in Loop: Header=BB12_100 Depth=1
	testl	%eax, %eax
	jne	.LBB12_115
# %bb.113:                              #   in Loop: Header=BB12_100 Depth=1
	movq	184(%rsp), %rax
	movq	168(%rsp), %rcx
	movq	160(%rsp), %rdx
	movq	176(%rsp), %rsi
	movq	%rax, 96(%rsp)
	movq	%rcx, 88(%rsp)
	movq	%rdx, 80(%rsp)
	movq	%rsi, 72(%rsp)
	movq	128(%rsp), %rax                 # 8-byte Reload
	movl	%eax, 12(%rsp)
	movq	120(%rsp), %rax                 # 8-byte Reload
	movl	%eax, 108(%rsp)
	movq	16(%rsp), %rax                  # 8-byte Reload
	movl	%eax, 104(%rsp)
	leaq	96(%rsp), %rax
	movq	%rax, 208(%rsp)
	leaq	88(%rsp), %rax
	movq	%rax, 216(%rsp)
	leaq	80(%rsp), %rax
	movq	%rax, 224(%rsp)
	leaq	72(%rsp), %rax
	movq	%rax, 232(%rsp)
	leaq	12(%rsp), %rax
	movq	%rax, 240(%rsp)
	leaq	108(%rsp), %rax
	movq	%rax, 248(%rsp)
	leaq	104(%rsp), %rax
	movq	%rax, 256(%rsp)
.Ltmp234:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	leaq	56(%rsp), %rdi
	leaq	40(%rsp), %rsi
	leaq	32(%rsp), %rdx
	leaq	24(%rsp), %rcx
	callq	__hipPopCallConfiguration@PLT
.Ltmp235:                               # EH_LABEL
# %bb.114:                              # %.noexc253
                                        #   in Loop: Header=BB12_100 Depth=1
	movq	56(%rsp), %rsi
	movl	64(%rsp), %edx
	movq	40(%rsp), %rcx
	movl	48(%rsp), %r8d
.Ltmp236:                               # EH_LABEL
	.cfi_escape 0x2e, 0x10
	movq	_Z21fused_m6_repacked_k64PK14__hip_fp8_e4m3PKhS3_Pfiii@GOTPCREL(%rip), %rdi
	movq	%rbp, %r9
	pushq	24(%rsp)
	.cfi_adjust_cfa_offset 8
	pushq	40(%rsp)
	.cfi_adjust_cfa_offset 8
	callq	hipLaunchKernel@PLT
	addq	$16, %rsp
	.cfi_adjust_cfa_offset -16
.Ltmp237:                               # EH_LABEL
	jmp	.LBB12_115
.LBB12_116:                             # %._crit_edge392
	movq	112(%rsp), %rdi
.Ltmp251:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	xorl	%esi, %esi
	callq	hipEventRecord@PLT
.Ltmp252:                               # EH_LABEL
# %bb.117:
	movq	112(%rsp), %rdi
.Ltmp253:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	callq	hipEventSynchronize@PLT
.Ltmp254:                               # EH_LABEL
# %bb.118:
	movq	144(%rsp), %rsi
	movq	112(%rsp), %rdx
.Ltmp256:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	leaq	208(%rsp), %rdi
	callq	hipEventElapsedTime@PLT
.Ltmp257:                               # EH_LABEL
# %bb.119:
	cvtsi2ssl	200(%rsp), %xmm0        # 4-byte Folded Reload
	movss	208(%rsp), %xmm1                # xmm1 = mem[0],zero,zero,zero
	divss	%xmm0, %xmm1
	movss	%xmm1, 208(%rsp)
	movq	184(%rsp), %rdi
.Ltmp258:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	callq	hipFree@PLT
.Ltmp259:                               # EH_LABEL
# %bb.120:
	movq	176(%rsp), %rdi
.Ltmp260:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	callq	hipFree@PLT
.Ltmp261:                               # EH_LABEL
# %bb.121:
	movq	168(%rsp), %rdi
.Ltmp262:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	callq	hipFree@PLT
.Ltmp263:                               # EH_LABEL
# %bb.122:
	movq	160(%rsp), %rdi
.Ltmp264:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	callq	hipFree@PLT
.Ltmp265:                               # EH_LABEL
# %bb.123:
	movq	144(%rsp), %rdi
.Ltmp266:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	callq	hipEventDestroy@PLT
.Ltmp267:                               # EH_LABEL
# %bb.124:
	movq	112(%rsp), %rdi
.Ltmp268:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	callq	hipEventDestroy@PLT
.Ltmp269:                               # EH_LABEL
# %bb.125:
	movss	208(%rsp), %xmm0                # xmm0 = mem[0],zero,zero,zero
	movss	%xmm0, 152(%rsp)                # 4-byte Spill
	movq	272(%rsp), %rdi                 # 8-byte Reload
	testq	%rdi, %rdi
	movq	288(%rsp), %r14                 # 8-byte Reload
	je	.LBB12_127
# %bb.126:
	movq	280(%rsp), %rsi                 # 8-byte Reload
	subq	%rdi, %rsi
	.cfi_escape 0x2e, 0x00
	callq	_ZdlPvm@PLT
.LBB12_127:                             # %_ZNSt6vectorIhSaIhEED2Ev.exit
	testq	%r14, %r14
	je	.LBB12_129
# %bb.128:
	movq	320(%rsp), %rsi                 # 8-byte Reload
	subq	%r14, %rsi
	.cfi_escape 0x2e, 0x00
	movq	%r14, %rdi
	callq	_ZdlPvm@PLT
.LBB12_129:                             # %_ZNSt6vectorIhSaIhEED2Ev.exit258
	.cfi_escape 0x2e, 0x00
	movl	$2048, %esi                     # imm = 0x800
	movq	%rbx, %rdi
	callq	_ZdlPvm@PLT
	testq	%r13, %r13
	je	.LBB12_131
# %bb.130:
	movq	328(%rsp), %rsi                 # 8-byte Reload
	subq	%r13, %rsi
	.cfi_escape 0x2e, 0x00
	movq	%r13, %rdi
	callq	_ZdlPvm@PLT
.LBB12_131:                             # %_ZNSt6vectorIhSaIhEED2Ev.exit260
	movss	152(%rsp), %xmm0                # 4-byte Reload
                                        # xmm0 = mem[0],zero,zero,zero
	cvtss2sd	%xmm0, %xmm0
	addq	$5432, %rsp                     # imm = 0x1538
	.cfi_def_cfa_offset 56
	popq	%rbx
	.cfi_def_cfa_offset 48
	popq	%r12
	.cfi_def_cfa_offset 40
	popq	%r13
	.cfi_def_cfa_offset 32
	popq	%r14
	.cfi_def_cfa_offset 24
	popq	%r15
	.cfi_def_cfa_offset 16
	popq	%rbp
	.cfi_def_cfa_offset 8
	retq
.LBB12_143:                             # %.noexc
	.cfi_def_cfa_offset 5488
	.cfi_escape 0x2e, 0x00
	leaq	.L.str.55(%rip), %rdi
	callq	_ZSt20__throw_length_errorPKc@PLT
.LBB12_13:
.Ltmp271:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	leaq	.L.str.55(%rip), %rdi
	callq	_ZSt20__throw_length_errorPKc@PLT
.Ltmp272:                               # EH_LABEL
# %bb.18:                               # %.noexc177
.LBB12_43:
.Ltmp185:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	leaq	.L.str.56(%rip), %rdi
	callq	_ZSt20__throw_length_errorPKc@PLT
.Ltmp186:                               # EH_LABEL
# %bb.44:                               # %.noexc269
.LBB12_58:
.Ltmp180:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	leaq	.L.str.56(%rip), %rdi
	callq	_ZSt20__throw_length_errorPKc@PLT
.Ltmp181:                               # EH_LABEL
# %bb.59:                               # %.noexc282
.LBB12_144:
.Ltmp182:                               # EH_LABEL
	jmp	.LBB12_54
.LBB12_53:
.Ltmp187:                               # EH_LABEL
.LBB12_54:                              # %_ZNSt6vectorIhSaIhEED2Ev.exit262
	movq	%rax, %r14
	movq	136(%rsp), %rbx                 # 8-byte Reload
	jmp	.LBB12_137
.LBB12_31:
.Ltmp166:                               # EH_LABEL
	movq	%rax, %r14
	jmp	.LBB12_140
.LBB12_34:
.Ltmp273:                               # EH_LABEL
	jmp	.LBB12_33
.LBB12_146:
.Ltmp255:                               # EH_LABEL
	jmp	.LBB12_133
.LBB12_132:
.Ltmp270:                               # EH_LABEL
	jmp	.LBB12_133
.LBB12_145:
.Ltmp202:                               # EH_LABEL
	movq	%rax, %r14
	jmp	.LBB12_135
.LBB12_39:
.Ltmp174:                               # EH_LABEL
	jmp	.LBB12_40
.LBB12_41:
.Ltmp177:                               # EH_LABEL
.LBB12_40:                              # %_ZNSt6vectorIhSaIhEED2Ev.exit262.thread
	movq	%rax, %r14
	movq	136(%rsp), %rbx                 # 8-byte Reload
	movq	%r12, %r15
	jmp	.LBB12_138
.LBB12_17:                              # %_ZNSt6vectorIhSaIhEED2Ev.exit266.thread
.Ltmp163:                               # EH_LABEL
	movq	%rax, %r14
	jmp	.LBB12_141
.LBB12_32:
.Ltmp169:                               # EH_LABEL
.LBB12_33:                              # %_ZNSt6vectorI14__hip_fp8_e4m3SaIS0_EED2Ev.exit264
	movq	%rax, %r14
	movq	136(%rsp), %rbx                 # 8-byte Reload
	jmp	.LBB12_139
.LBB12_105:
.Ltmp250:                               # EH_LABEL
.LBB12_133:
	movq	%rax, %r14
	movq	304(%rsp), %r13                 # 8-byte Reload
	movq	136(%rsp), %rbx                 # 8-byte Reload
	jmp	.LBB12_134
.LBB12_83:
.Ltmp225:                               # EH_LABEL
	movq	%rax, %r14
.LBB12_134:
	movq	288(%rsp), %r15                 # 8-byte Reload
.LBB12_135:
	movq	272(%rsp), %rdi                 # 8-byte Reload
	testq	%rdi, %rdi
	je	.LBB12_137
# %bb.136:
	movq	280(%rsp), %rsi                 # 8-byte Reload
	subq	%rdi, %rsi
	.cfi_escape 0x2e, 0x00
	callq	_ZdlPvm@PLT
.LBB12_137:                             # %_ZNSt6vectorIhSaIhEED2Ev.exit262
	testq	%r15, %r15
	je	.LBB12_139
.LBB12_138:                             # %_ZNSt6vectorIhSaIhEED2Ev.exit262.thread
	movq	320(%rsp), %rsi                 # 8-byte Reload
	subq	%r15, %rsi
	.cfi_escape 0x2e, 0x00
	movq	%r15, %rdi
	callq	_ZdlPvm@PLT
.LBB12_139:                             # %_ZNSt6vectorI14__hip_fp8_e4m3SaIS0_EED2Ev.exit264
	.cfi_escape 0x2e, 0x00
	movl	$2048, %esi                     # imm = 0x800
	movq	%rbx, %rdi
	callq	_ZdlPvm@PLT
.LBB12_140:                             # %_ZNSt6vectorIhSaIhEED2Ev.exit266
	testq	%r13, %r13
	je	.LBB12_142
.LBB12_141:
	movq	328(%rsp), %rsi                 # 8-byte Reload
	subq	%r13, %rsi
	.cfi_escape 0x2e, 0x00
	movq	%r13, %rdi
	callq	_ZdlPvm@PLT
.LBB12_142:                             # %_ZNSt6vectorIhSaIhEED2Ev.exit268
	.cfi_escape 0x2e, 0x00
	movq	%r14, %rdi
	callq	_Unwind_Resume@PLT
.Lfunc_end12:
	.size	_Z14bench_fused_m6iiiii, .Lfunc_end12-_Z14bench_fused_m6iiiii
	.cfi_endproc
	.section	.gcc_except_table,"a",@progbits
	.p2align	2, 0x0
GCC_except_table12:
.Lexception2:
	.byte	255                             # @LPStart Encoding = omit
	.byte	255                             # @TType Encoding = omit
	.byte	1                               # Call site Encoding = uleb128
	.uleb128 .Lcst_end2-.Lcst_begin2
.Lcst_begin2:
	.uleb128 .Lfunc_begin2-.Lfunc_begin2    # >> Call Site 1 <<
	.uleb128 .Ltmp161-.Lfunc_begin2         #   Call between .Lfunc_begin2 and .Ltmp161
	.byte	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp161-.Lfunc_begin2         # >> Call Site 2 <<
	.uleb128 .Ltmp162-.Ltmp161              #   Call between .Ltmp161 and .Ltmp162
	.uleb128 .Ltmp163-.Lfunc_begin2         #     jumps to .Ltmp163
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp164-.Lfunc_begin2         # >> Call Site 3 <<
	.uleb128 .Ltmp165-.Ltmp164              #   Call between .Ltmp164 and .Ltmp165
	.uleb128 .Ltmp166-.Lfunc_begin2         #     jumps to .Ltmp166
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp165-.Lfunc_begin2         # >> Call Site 4 <<
	.uleb128 .Ltmp167-.Ltmp165              #   Call between .Ltmp165 and .Ltmp167
	.byte	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp167-.Lfunc_begin2         # >> Call Site 5 <<
	.uleb128 .Ltmp168-.Ltmp167              #   Call between .Ltmp167 and .Ltmp168
	.uleb128 .Ltmp169-.Lfunc_begin2         #     jumps to .Ltmp169
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp170-.Lfunc_begin2         # >> Call Site 6 <<
	.uleb128 .Ltmp171-.Ltmp170              #   Call between .Ltmp170 and .Ltmp171
	.uleb128 .Ltmp273-.Lfunc_begin2         #     jumps to .Ltmp273
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp171-.Lfunc_begin2         # >> Call Site 7 <<
	.uleb128 .Ltmp172-.Ltmp171              #   Call between .Ltmp171 and .Ltmp172
	.byte	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp172-.Lfunc_begin2         # >> Call Site 8 <<
	.uleb128 .Ltmp173-.Ltmp172              #   Call between .Ltmp172 and .Ltmp173
	.uleb128 .Ltmp174-.Lfunc_begin2         #     jumps to .Ltmp174
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp175-.Lfunc_begin2         # >> Call Site 9 <<
	.uleb128 .Ltmp176-.Ltmp175              #   Call between .Ltmp175 and .Ltmp176
	.uleb128 .Ltmp177-.Lfunc_begin2         #     jumps to .Ltmp177
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp183-.Lfunc_begin2         # >> Call Site 10 <<
	.uleb128 .Ltmp184-.Ltmp183              #   Call between .Ltmp183 and .Ltmp184
	.uleb128 .Ltmp187-.Lfunc_begin2         #     jumps to .Ltmp187
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp184-.Lfunc_begin2         # >> Call Site 11 <<
	.uleb128 .Ltmp178-.Ltmp184              #   Call between .Ltmp184 and .Ltmp178
	.byte	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp178-.Lfunc_begin2         # >> Call Site 12 <<
	.uleb128 .Ltmp179-.Ltmp178              #   Call between .Ltmp178 and .Ltmp179
	.uleb128 .Ltmp182-.Lfunc_begin2         #     jumps to .Ltmp182
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp179-.Lfunc_begin2         # >> Call Site 13 <<
	.uleb128 .Ltmp188-.Ltmp179              #   Call between .Ltmp179 and .Ltmp188
	.byte	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp188-.Lfunc_begin2         # >> Call Site 14 <<
	.uleb128 .Ltmp201-.Ltmp188              #   Call between .Ltmp188 and .Ltmp201
	.uleb128 .Ltmp202-.Lfunc_begin2         #     jumps to .Ltmp202
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp203-.Lfunc_begin2         # >> Call Site 15 <<
	.uleb128 .Ltmp224-.Ltmp203              #   Call between .Ltmp203 and .Ltmp224
	.uleb128 .Ltmp225-.Lfunc_begin2         #     jumps to .Ltmp225
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp226-.Lfunc_begin2         # >> Call Site 16 <<
	.uleb128 .Ltmp231-.Ltmp226              #   Call between .Ltmp226 and .Ltmp231
	.uleb128 .Ltmp255-.Lfunc_begin2         #     jumps to .Ltmp255
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp238-.Lfunc_begin2         # >> Call Site 17 <<
	.uleb128 .Ltmp237-.Ltmp238              #   Call between .Ltmp238 and .Ltmp237
	.uleb128 .Ltmp250-.Lfunc_begin2         #     jumps to .Ltmp250
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp251-.Lfunc_begin2         # >> Call Site 18 <<
	.uleb128 .Ltmp254-.Ltmp251              #   Call between .Ltmp251 and .Ltmp254
	.uleb128 .Ltmp255-.Lfunc_begin2         #     jumps to .Ltmp255
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp256-.Lfunc_begin2         # >> Call Site 19 <<
	.uleb128 .Ltmp269-.Ltmp256              #   Call between .Ltmp256 and .Ltmp269
	.uleb128 .Ltmp270-.Lfunc_begin2         #     jumps to .Ltmp270
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp269-.Lfunc_begin2         # >> Call Site 20 <<
	.uleb128 .Ltmp271-.Ltmp269              #   Call between .Ltmp269 and .Ltmp271
	.byte	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp271-.Lfunc_begin2         # >> Call Site 21 <<
	.uleb128 .Ltmp272-.Ltmp271              #   Call between .Ltmp271 and .Ltmp272
	.uleb128 .Ltmp273-.Lfunc_begin2         #     jumps to .Ltmp273
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp185-.Lfunc_begin2         # >> Call Site 22 <<
	.uleb128 .Ltmp186-.Ltmp185              #   Call between .Ltmp185 and .Ltmp186
	.uleb128 .Ltmp187-.Lfunc_begin2         #     jumps to .Ltmp187
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp180-.Lfunc_begin2         # >> Call Site 23 <<
	.uleb128 .Ltmp181-.Ltmp180              #   Call between .Ltmp180 and .Ltmp181
	.uleb128 .Ltmp182-.Lfunc_begin2         #     jumps to .Ltmp182
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp181-.Lfunc_begin2         # >> Call Site 24 <<
	.uleb128 .Lfunc_end12-.Ltmp181          #   Call between .Ltmp181 and .Lfunc_end12
	.byte	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
.Lcst_end2:
	.p2align	2, 0x0
                                        # -- End function
	.text
	.globl	_Z36__device_stub__decode_only_scatteredPKhS0_Phiii # -- Begin function _Z36__device_stub__decode_only_scatteredPKhS0_Phiii
	.prefalign	4, .Lfunc_end13, nop
	.type	_Z36__device_stub__decode_only_scatteredPKhS0_Phiii,@function
_Z36__device_stub__decode_only_scatteredPKhS0_Phiii: # @_Z36__device_stub__decode_only_scatteredPKhS0_Phiii
	.cfi_startproc
# %bb.0:
	subq	$152, %rsp
	.cfi_def_cfa_offset 160
	movq	%rdi, 88(%rsp)
	movq	%rsi, 80(%rsp)
	movq	%rdx, 72(%rsp)
	movl	%ecx, 20(%rsp)
	movl	%r8d, 16(%rsp)
	movl	%r9d, 12(%rsp)
	leaq	88(%rsp), %rax
	movq	%rax, 96(%rsp)
	leaq	80(%rsp), %rax
	movq	%rax, 104(%rsp)
	leaq	72(%rsp), %rax
	movq	%rax, 112(%rsp)
	leaq	20(%rsp), %rax
	movq	%rax, 120(%rsp)
	leaq	16(%rsp), %rax
	movq	%rax, 128(%rsp)
	leaq	12(%rsp), %rax
	movq	%rax, 136(%rsp)
	leaq	56(%rsp), %rdi
	leaq	40(%rsp), %rsi
	leaq	32(%rsp), %rdx
	leaq	24(%rsp), %rcx
	callq	__hipPopCallConfiguration@PLT
	movq	56(%rsp), %rsi
	movl	64(%rsp), %edx
	movq	40(%rsp), %rcx
	movl	48(%rsp), %r8d
	movq	_Z21decode_only_scatteredPKhS0_Phiii@GOTPCREL(%rip), %rdi
	leaq	96(%rsp), %r9
	pushq	24(%rsp)
	.cfi_adjust_cfa_offset 8
	pushq	40(%rsp)
	.cfi_adjust_cfa_offset 8
	callq	hipLaunchKernel@PLT
	addq	$168, %rsp
	.cfi_adjust_cfa_offset -168
	retq
.Lfunc_end13:
	.size	_Z36__device_stub__decode_only_scatteredPKhS0_Phiii, .Lfunc_end13-_Z36__device_stub__decode_only_scatteredPKhS0_Phiii
	.cfi_endproc
                                        # -- End function
	.globl	_Z35__device_stub__decode_only_repackedPKhS0_Phii # -- Begin function _Z35__device_stub__decode_only_repackedPKhS0_Phii
	.prefalign	4, .Lfunc_end14, nop
	.type	_Z35__device_stub__decode_only_repackedPKhS0_Phii,@function
_Z35__device_stub__decode_only_repackedPKhS0_Phii: # @_Z35__device_stub__decode_only_repackedPKhS0_Phii
	.cfi_startproc
# %bb.0:
	subq	$120, %rsp
	.cfi_def_cfa_offset 128
	movq	%rdi, 72(%rsp)
	movq	%rsi, 64(%rsp)
	movq	%rdx, 56(%rsp)
	movl	%ecx, 4(%rsp)
	movl	%r8d, (%rsp)
	leaq	72(%rsp), %rax
	movq	%rax, 80(%rsp)
	leaq	64(%rsp), %rax
	movq	%rax, 88(%rsp)
	leaq	56(%rsp), %rax
	movq	%rax, 96(%rsp)
	leaq	4(%rsp), %rax
	movq	%rax, 104(%rsp)
	movq	%rsp, %rax
	movq	%rax, 112(%rsp)
	leaq	40(%rsp), %rdi
	leaq	24(%rsp), %rsi
	leaq	16(%rsp), %rdx
	leaq	8(%rsp), %rcx
	callq	__hipPopCallConfiguration@PLT
	movq	40(%rsp), %rsi
	movl	48(%rsp), %edx
	movq	24(%rsp), %rcx
	movl	32(%rsp), %r8d
	movq	_Z20decode_only_repackedPKhS0_Phii@GOTPCREL(%rip), %rdi
	leaq	80(%rsp), %r9
	pushq	8(%rsp)
	.cfi_adjust_cfa_offset 8
	pushq	24(%rsp)
	.cfi_adjust_cfa_offset 8
	callq	hipLaunchKernel@PLT
	addq	$136, %rsp
	.cfi_adjust_cfa_offset -136
	retq
.Lfunc_end14:
	.size	_Z35__device_stub__decode_only_repackedPKhS0_Phii, .Lfunc_end14-_Z35__device_stub__decode_only_repackedPKhS0_Phii
	.cfi_endproc
                                        # -- End function
	.globl	_Z21bench_decode_isolatediibi   # -- Begin function _Z21bench_decode_isolatediibi
	.prefalign	4, .Lfunc_end15, nop
	.type	_Z21bench_decode_isolatediibi,@function
_Z21bench_decode_isolatediibi:          # @_Z21bench_decode_isolatediibi
.Lfunc_begin3:
	.cfi_startproc
	.cfi_personality 155, DW.ref.__gxx_personality_v0
	.cfi_lsda 27, .Lexception3
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	pushq	%r15
	.cfi_def_cfa_offset 24
	pushq	%r14
	.cfi_def_cfa_offset 32
	pushq	%r13
	.cfi_def_cfa_offset 40
	pushq	%r12
	.cfi_def_cfa_offset 48
	pushq	%rbx
	.cfi_def_cfa_offset 56
	subq	$5368, %rsp                     # imm = 0x14F8
	.cfi_def_cfa_offset 5424
	.cfi_offset %rbx, -56
	.cfi_offset %r12, -48
	.cfi_offset %r13, -40
	.cfi_offset %r14, -32
	.cfi_offset %r15, -24
	.cfi_offset %rbp, -16
	movl	%edx, 144(%rsp)                 # 4-byte Spill
                                        # kill: def $esi killed $esi def $rsi
                                        # kill: def $edi killed $edi def $rdi
	movq	%rdi, 24(%rsp)                  # 8-byte Spill
	leal	255(%rsi), %edi
	testl	%esi, %esi
	movq	%rsi, 32(%rsp)                  # 8-byte Spill
	cmovnsl	%esi, %edi
	sarl	$8, %edi
	movq	$2, 368(%rsp)
	movl	$2, %esi
	movl	$2, %eax
	.p2align	4
.LBB15_1:                               # =>This Inner Loop Header: Depth=1
	movq	%rsi, %rdx
	shrq	$30, %rdx
	xorq	%rsi, %rdx
	imulq	$1812433253, %rdx, %rdx         # imm = 0x6C078965
	leaq	(%rax,%rdx), %rsi
	decq	%rsi
	movl	%esi, %edx
	movq	%rdx, 360(%rsp,%rax,8)
	cmpq	$624, %rax                      # imm = 0x270
	je	.LBB15_3
# %bb.2:                                #   in Loop: Header=BB15_1 Depth=1
	shrl	$30, %edx
	xorl	%edx, %esi
	imull	$1812433253, %esi, %esi         # imm = 0x6C078965
	addl	%eax, %esi
	movq	%rsi, 368(%rsp,%rax,8)
	addq	$2, %rax
	jmp	.LBB15_1
.LBB15_3:                               # %_ZNSt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EEC2Em.exit
	movl	%edi, %eax
	shll	$7, %eax
	movq	$624, 5360(%rsp)                # imm = 0x270
	movl	%eax, 140(%rsp)                 # 4-byte Spill
	imull	24(%rsp), %eax                  # 4-byte Folded Reload
	testl	%eax, %eax
	js	.LBB15_72
# %bb.4:                                # %_ZNSt6vectorIhSaIhEE17_S_check_init_lenEmRKS0_.exit.i
	movl	%ecx, 256(%rsp)                 # 4-byte Spill
	movslq	%eax, %r14
	je	.LBB15_8
# %bb.5:                                # %_ZNSt6vectorIhSaIhEEC2EmRKS0_.exit
	.cfi_escape 0x2e, 0x00
	movq	%r14, %rdi
	callq	_Znwm@PLT
	movq	%rax, %r12
	leaq	(%rax,%r14), %r13
	movb	$0, (%rax)
	leaq	1(%rax), %rdi
	leaq	-1(%r14), %rdx
	.cfi_escape 0x2e, 0x00
	xorl	%ebx, %ebx
	xorl	%esi, %esi
	callq	memset@PLT
	leaq	368(%rsp), %r15
	.p2align	4
.LBB15_6:                               # %.lr.ph
                                        # =>This Inner Loop Header: Depth=1
.Ltmp274:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movq	%r15, %rdi
	callq	_ZNSt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EEclEv
.Ltmp275:                               # EH_LABEL
# %bb.7:                                #   in Loop: Header=BB15_6 Depth=1
	movb	%al, (%r12,%rbx)
	incq	%rbx
	cmpq	%rbx, %r14
	jne	.LBB15_6
	jmp	.LBB15_9
.LBB15_8:
	xorl	%r12d, %r12d
	xorl	%r13d, %r13d
.LBB15_9:                               # %._crit_edge
.Ltmp277:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movl	$2048, %edi                     # imm = 0x800
	callq	_Znwm@PLT
.Ltmp278:                               # EH_LABEL
# %bb.10:
	movq	%rax, %r15
	.cfi_escape 0x2e, 0x00
	xorl	%ebx, %ebx
	movl	$2048, %edx                     # imm = 0x800
	movq	%rax, %rdi
	xorl	%esi, %esi
	callq	memset@PLT
	leaq	368(%rsp), %rbp
	.p2align	4
.LBB15_11:                              # =>This Inner Loop Header: Depth=1
.Ltmp280:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movq	%rbp, %rdi
	callq	_ZNSt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EEclEv
.Ltmp281:                               # EH_LABEL
# %bb.12:                               #   in Loop: Header=BB15_11 Depth=1
	movb	%al, (%r15,%rbx)
	incq	%rbx
	cmpq	$2048, %rbx                     # imm = 0x800
	jne	.LBB15_11
# %bb.13:
.Ltmp283:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	leaq	160(%rsp), %rdi
	movl	$2048, %esi                     # imm = 0x800
	callq	hipMalloc@PLT
.Ltmp284:                               # EH_LABEL
# %bb.14:                               # %_ZL9hipMallocIhE10hipError_tPPT_m.exit
	movq	160(%rsp), %rdi
.Ltmp285:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movl	$2048, %edx                     # imm = 0x800
	movq	%r15, %rsi
	movl	$1, %ecx
	callq	hipMemcpy@PLT
.Ltmp286:                               # EH_LABEL
# %bb.15:
	movslq	24(%rsp), %rax                  # 4-byte Folded Reload
	movslq	32(%rsp), %rsi                  # 4-byte Folded Reload
	imulq	%rax, %rsi
.Ltmp287:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	leaq	168(%rsp), %rdi
	callq	hipMalloc@PLT
.Ltmp288:                               # EH_LABEL
# %bb.16:                               # %_ZL9hipMallocIhE10hipError_tPPT_m.exit114
	cmpb	$0, 144(%rsp)                   # 1-byte Folded Reload
	movq	%r12, 232(%rsp)                 # 8-byte Spill
	movq	%r15, 224(%rsp)                 # 8-byte Spill
	je	.LBB15_21
# %bb.17:
	movq	24(%rsp), %rcx                  # 8-byte Reload
	leal	15(%rcx), %eax
	testl	%ecx, %ecx
	cmovnsl	%ecx, %eax
	sarl	$4, %eax
	movq	32(%rsp), %rcx                  # 8-byte Reload
	leal	31(%rcx), %ebx
	testl	%ecx, %ecx
	cmovnsl	%ecx, %ebx
	sarl	$5, %ebx
	imull	%ebx, %eax
	movl	%eax, %ecx
	shll	$8, %ecx
	movslq	%ecx, %rbp
	testl	%eax, %eax
	movq	%r13, 280(%rsp)                 # 8-byte Spill
	movq	%rbp, 272(%rsp)                 # 8-byte Spill
	je	.LBB15_24
# %bb.18:
	js	.LBB15_73
# %bb.19:
	movq	$0, 16(%rsp)                    # 8-byte Folded Spill
.Ltmp294:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movq	$0, 152(%rsp)                   # 8-byte Folded Spill
	movq	%rbp, %rdi
	callq	_Znwm@PLT
.Ltmp295:                               # EH_LABEL
# %bb.20:                               # %.noexc115
	movq	%rax, %r14
	movb	$0, (%rax)
	leaq	-1(%rbp), %rdx
	leaq	1(%rax), %rdi
	.cfi_escape 0x2e, 0x00
	xorl	%esi, %esi
	callq	memset@PLT
	movq	%r14, %rax
	addq	%rbp, %rax
	movq	%rax, 152(%rsp)                 # 8-byte Spill
	movq	%r14, 16(%rsp)                  # 8-byte Spill
	cmpl	$16, 24(%rsp)                   # 4-byte Folded Reload
	jge	.LBB15_25
	jmp	.LBB15_30
.LBB15_21:
.Ltmp289:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	leaq	48(%rsp), %rdi
	movq	%r14, %rsi
	callq	hipMalloc@PLT
.Ltmp290:                               # EH_LABEL
# %bb.22:                               # %_ZL9hipMallocIhE10hipError_tPPT_m.exit119
	movq	48(%rsp), %rdi
.Ltmp291:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movq	%r12, %rsi
	movq	%r14, %rdx
	movl	$1, %ecx
	callq	hipMemcpy@PLT
.Ltmp292:                               # EH_LABEL
# %bb.23:
	movq	$0, 16(%rsp)                    # 8-byte Folded Spill
	movq	$0, 152(%rsp)                   # 8-byte Folded Spill
	jmp	.LBB15_32
.LBB15_24:
	movq	$0, 16(%rsp)                    # 8-byte Folded Spill
	movq	$0, 152(%rsp)                   # 8-byte Folded Spill
	cmpl	$16, 24(%rsp)                   # 4-byte Folded Reload
	jl	.LBB15_30
.LBB15_25:                              # %.preheader.lr.ph.i
	movq	24(%rsp), %rax                  # 8-byte Reload
	movl	%eax, %ecx
	shrl	$4, %ecx
	movslq	140(%rsp), %rax                 # 4-byte Folded Reload
	movq	%rax, 296(%rsp)                 # 8-byte Spill
	movl	%ebx, %eax
	movq	%rax, 288(%rsp)                 # 8-byte Spill
	movq	16(%rsp), %rax                  # 8-byte Reload
	addq	$240, %rax
	movq	%rax, 264(%rsp)                 # 8-byte Spill
	movq	%rcx, 304(%rsp)                 # 8-byte Spill
	movq	%rcx, %r15
	shlq	$8, %r15
	movq	$0, 240(%rsp)                   # 8-byte Folded Spill
	jmp	.LBB15_27
	.p2align	4
.LBB15_26:                              # %._crit_edge.i
                                        #   in Loop: Header=BB15_27 Depth=1
	movq	240(%rsp), %rcx                 # 8-byte Reload
	incq	%rcx
	addq	$256, 264(%rsp)                 # 8-byte Folded Spill
                                        # imm = 0x100
	movq	%rcx, 240(%rsp)                 # 8-byte Spill
	cmpq	304(%rsp), %rcx                 # 8-byte Folded Reload
	je	.LBB15_30
.LBB15_27:                              # %.preheader.i
                                        # =>This Loop Header: Depth=1
                                        #     Child Loop BB15_29 Depth 2
	cmpl	$32, 32(%rsp)                   # 4-byte Folded Reload
	movq	232(%rsp), %r13                 # 8-byte Reload
	jl	.LBB15_26
# %bb.28:                               # %.lr.ph.i
                                        #   in Loop: Header=BB15_27 Depth=1
	movq	240(%rsp), %r9                  # 8-byte Reload
	shlq	$4, %r9
	movq	%r9, %rax
	movq	296(%rsp), %rdx                 # 8-byte Reload
	imulq	%rdx, %rax
	movq	%rax, 248(%rsp)                 # 8-byte Spill
	leaq	1(%r9), %rax
	imulq	%rdx, %rax
	movq	%rax, 360(%rsp)                 # 8-byte Spill
	leaq	2(%r9), %rax
	imulq	%rdx, %rax
	movq	%rax, 352(%rsp)                 # 8-byte Spill
	leaq	3(%r9), %rax
	imulq	%rdx, %rax
	movq	%rax, 344(%rsp)                 # 8-byte Spill
	leaq	4(%r9), %rax
	imulq	%rdx, %rax
	movq	%rax, 336(%rsp)                 # 8-byte Spill
	leaq	5(%r9), %rax
	imulq	%rdx, %rax
	movq	%rax, 328(%rsp)                 # 8-byte Spill
	leaq	6(%r9), %rax
	imulq	%rdx, %rax
	movq	%rax, 320(%rsp)                 # 8-byte Spill
	leaq	7(%r9), %rax
	imulq	%rdx, %rax
	movq	%rax, 312(%rsp)                 # 8-byte Spill
	leaq	8(%r9), %r12
	imulq	%rdx, %r12
	leaq	9(%r9), %rbx
	imulq	%rdx, %rbx
	leaq	10(%r9), %rsi
	imulq	%rdx, %rsi
	leaq	11(%r9), %r8
	imulq	%rdx, %r8
	leaq	12(%r9), %rcx
	imulq	%rdx, %rcx
	leaq	13(%r9), %rdi
	imulq	%rdx, %rdi
	leaq	14(%r9), %r10
	imulq	%rdx, %r10
	orq	$15, %r9
	imulq	%rdx, %r9
	movq	264(%rsp), %r11                 # 8-byte Reload
	movq	288(%rsp), %r14                 # 8-byte Reload
	xorl	%edx, %edx
	.p2align	4
.LBB15_29:                              #   Parent Loop BB15_27 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	movl	%edx, %ebp
	andl	$112, %ebp
	addq	%r13, %rbp
	movq	%r13, %rax
	movq	%r15, %r13
	movl	%edx, %r15d
	andl	$1073741696, %r15d              # imm = 0x3FFFFF80
	addq	%rbp, %r15
	movq	248(%rsp), %rbp                 # 8-byte Reload
	movups	(%rbp,%r15), %xmm0
	movups	%xmm0, -240(%r11)
	movq	360(%rsp), %rbp                 # 8-byte Reload
	movups	(%rbp,%r15), %xmm0
	movups	%xmm0, -224(%r11)
	movq	352(%rsp), %rbp                 # 8-byte Reload
	movups	(%rbp,%r15), %xmm0
	movups	%xmm0, -208(%r11)
	movq	344(%rsp), %rbp                 # 8-byte Reload
	movups	(%rbp,%r15), %xmm0
	movups	%xmm0, -192(%r11)
	movq	336(%rsp), %rbp                 # 8-byte Reload
	movups	(%rbp,%r15), %xmm0
	movups	%xmm0, -176(%r11)
	movq	328(%rsp), %rbp                 # 8-byte Reload
	movups	(%rbp,%r15), %xmm0
	movups	%xmm0, -160(%r11)
	movq	320(%rsp), %rbp                 # 8-byte Reload
	movups	(%rbp,%r15), %xmm0
	movups	%xmm0, -144(%r11)
	movq	312(%rsp), %rbp                 # 8-byte Reload
	movups	(%rbp,%r15), %xmm0
	movups	%xmm0, -128(%r11)
	movups	(%r12,%r15), %xmm0
	movups	%xmm0, -112(%r11)
	movups	(%rbx,%r15), %xmm0
	movups	%xmm0, -96(%r11)
	movups	(%rsi,%r15), %xmm0
	movups	%xmm0, -80(%r11)
	movups	(%r8,%r15), %xmm0
	movups	%xmm0, -64(%r11)
	movups	(%rcx,%r15), %xmm0
	movups	%xmm0, -48(%r11)
	movups	(%rdi,%r15), %xmm0
	movups	%xmm0, -32(%r11)
	movups	(%r10,%r15), %xmm0
	movups	%xmm0, -16(%r11)
	movups	(%r9,%r15), %xmm0
	movq	%r13, %r15
	movq	%rax, %r13
	movups	%xmm0, (%r11)
	addq	$16, %rdx
	addq	%r15, %r11
	decq	%r14
	jne	.LBB15_29
	jmp	.LBB15_26
.LBB15_30:                              # %_Z17repack_k32_for_m6PKhPhiii.exit
.Ltmp298:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	leaq	48(%rsp), %rdi
	movq	272(%rsp), %rbx                 # 8-byte Reload
	movq	%rbx, %rsi
	movq	232(%rsp), %r12                 # 8-byte Reload
	movq	280(%rsp), %r13                 # 8-byte Reload
	callq	hipMalloc@PLT
.Ltmp299:                               # EH_LABEL
# %bb.31:                               # %_ZL9hipMallocIhE10hipError_tPPT_m.exit117
	movq	48(%rsp), %rdi
.Ltmp300:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movq	16(%rsp), %rsi                  # 8-byte Reload
	movq	%rbx, %rdx
	movl	$1, %ecx
	callq	hipMemcpy@PLT
.Ltmp301:                               # EH_LABEL
.LBB15_32:
	movq	32(%rsp), %rcx                  # 8-byte Reload
	leal	7(%rcx), %eax
	testl	%ecx, %ecx
	cmovnsl	%ecx, %eax
	sarl	$3, %eax
	imull	24(%rsp), %eax                  # 4-byte Folded Reload
	leal	255(%rax), %ecx
	addl	$510, %eax                      # imm = 0x1FE
	testl	%ecx, %ecx
	cmovnsl	%ecx, %eax
	sarl	$8, %eax
	movabsq	$4294967296, %rbp               # imm = 0x100000000
	orq	%rax, %rbp
	movl	$20, %r15d
	movabsq	$4294967552, %r14               # imm = 0x100000100
	movl	256(%rsp), %ebx                 # 4-byte Reload
	.p2align	4
.LBB15_33:                              # =>This Inner Loop Header: Depth=1
	cmpb	$0, 144(%rsp)                   # 1-byte Folded Reload
	je	.LBB15_38
# %bb.34:                               #   in Loop: Header=BB15_33 Depth=1
.Ltmp309:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movq	%rbp, %rdi
	movl	$1, %esi
	movq	%r14, %rdx
	movl	$1, %ecx
	xorl	%r8d, %r8d
	xorl	%r9d, %r9d
	callq	__hipPushCallConfiguration@PLT
.Ltmp310:                               # EH_LABEL
# %bb.35:                               #   in Loop: Header=BB15_33 Depth=1
	testl	%eax, %eax
	jne	.LBB15_42
# %bb.36:                               #   in Loop: Header=BB15_33 Depth=1
	movq	48(%rsp), %rax
	movq	160(%rsp), %rcx
	movq	168(%rsp), %rdx
	movq	%rax, 128(%rsp)
	movq	%rcx, 120(%rsp)
	movq	%rdx, 112(%rsp)
	movq	24(%rsp), %rax                  # 8-byte Reload
	movl	%eax, 56(%rsp)
	movq	32(%rsp), %rax                  # 8-byte Reload
	movl	%eax, 40(%rsp)
	leaq	128(%rsp), %rax
	movq	%rax, 176(%rsp)
	leaq	120(%rsp), %rax
	movq	%rax, 184(%rsp)
	leaq	112(%rsp), %rax
	movq	%rax, 192(%rsp)
	leaq	56(%rsp), %rax
	movq	%rax, 200(%rsp)
	leaq	40(%rsp), %rax
	movq	%rax, 208(%rsp)
.Ltmp311:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	leaq	96(%rsp), %rdi
	leaq	80(%rsp), %rsi
	leaq	72(%rsp), %rdx
	leaq	64(%rsp), %rcx
	callq	__hipPopCallConfiguration@PLT
.Ltmp312:                               # EH_LABEL
# %bb.37:                               # %.noexc120
                                        #   in Loop: Header=BB15_33 Depth=1
	movq	96(%rsp), %rsi
	movl	104(%rsp), %edx
	movq	80(%rsp), %rcx
	movl	88(%rsp), %r8d
.Ltmp313:                               # EH_LABEL
	.cfi_escape 0x2e, 0x10
	movq	_Z20decode_only_repackedPKhS0_Phii@GOTPCREL(%rip), %rdi
	leaq	176(%rsp), %r9
	pushq	64(%rsp)
	.cfi_adjust_cfa_offset 8
	pushq	80(%rsp)
	.cfi_adjust_cfa_offset 8
	callq	hipLaunchKernel@PLT
	addq	$16, %rsp
	.cfi_adjust_cfa_offset -16
.Ltmp314:                               # EH_LABEL
	jmp	.LBB15_42
	.p2align	4
.LBB15_38:                              #   in Loop: Header=BB15_33 Depth=1
.Ltmp303:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movq	%rbp, %rdi
	movl	$1, %esi
	movq	%r14, %rdx
	movl	$1, %ecx
	xorl	%r8d, %r8d
	xorl	%r9d, %r9d
	callq	__hipPushCallConfiguration@PLT
.Ltmp304:                               # EH_LABEL
# %bb.39:                               #   in Loop: Header=BB15_33 Depth=1
	testl	%eax, %eax
	jne	.LBB15_42
# %bb.40:                               #   in Loop: Header=BB15_33 Depth=1
	movq	48(%rsp), %rax
	movq	160(%rsp), %rcx
	movq	168(%rsp), %rdx
	movq	%rax, 128(%rsp)
	movq	%rcx, 120(%rsp)
	movq	%rdx, 112(%rsp)
	movq	24(%rsp), %rax                  # 8-byte Reload
	movl	%eax, 56(%rsp)
	movq	32(%rsp), %rax                  # 8-byte Reload
	movl	%eax, 40(%rsp)
	movl	140(%rsp), %eax                 # 4-byte Reload
	movl	%eax, 12(%rsp)
	leaq	128(%rsp), %rax
	movq	%rax, 176(%rsp)
	leaq	120(%rsp), %rax
	movq	%rax, 184(%rsp)
	leaq	112(%rsp), %rax
	movq	%rax, 192(%rsp)
	leaq	56(%rsp), %rax
	movq	%rax, 200(%rsp)
	leaq	40(%rsp), %rax
	movq	%rax, 208(%rsp)
	leaq	12(%rsp), %rax
	movq	%rax, 216(%rsp)
.Ltmp305:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	leaq	96(%rsp), %rdi
	leaq	80(%rsp), %rsi
	leaq	72(%rsp), %rdx
	leaq	64(%rsp), %rcx
	callq	__hipPopCallConfiguration@PLT
.Ltmp306:                               # EH_LABEL
# %bb.41:                               # %.noexc128
                                        #   in Loop: Header=BB15_33 Depth=1
	movq	96(%rsp), %rsi
	movl	104(%rsp), %edx
	movq	80(%rsp), %rcx
	movl	88(%rsp), %r8d
.Ltmp307:                               # EH_LABEL
	.cfi_escape 0x2e, 0x10
	movq	_Z21decode_only_scatteredPKhS0_Phiii@GOTPCREL(%rip), %rdi
	leaq	176(%rsp), %r9
	pushq	64(%rsp)
	.cfi_adjust_cfa_offset 8
	pushq	80(%rsp)
	.cfi_adjust_cfa_offset 8
	callq	hipLaunchKernel@PLT
	addq	$16, %rsp
	.cfi_adjust_cfa_offset -16
.Ltmp308:                               # EH_LABEL
	.p2align	4
.LBB15_42:                              #   in Loop: Header=BB15_33 Depth=1
.Ltmp315:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	callq	hipDeviceSynchronize@PLT
.Ltmp316:                               # EH_LABEL
# %bb.43:                               #   in Loop: Header=BB15_33 Depth=1
	decl	%r15d
	jne	.LBB15_33
# %bb.44:
.Ltmp318:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	leaq	56(%rsp), %rdi
	callq	hipEventCreate@PLT
.Ltmp319:                               # EH_LABEL
# %bb.45:
.Ltmp320:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	leaq	40(%rsp), %rdi
	callq	hipEventCreate@PLT
.Ltmp321:                               # EH_LABEL
# %bb.46:
	movq	56(%rsp), %rdi
.Ltmp322:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	xorl	%esi, %esi
	callq	hipEventRecord@PLT
.Ltmp323:                               # EH_LABEL
# %bb.47:                               # %.preheader
	testl	%ebx, %ebx
	jle	.LBB15_59
# %bb.48:
	movabsq	$4294967552, %r14               # imm = 0x100000100
	movl	%ebx, %r15d
	jmp	.LBB15_50
	.p2align	4
.LBB15_49:                              #   in Loop: Header=BB15_50 Depth=1
	decl	%r15d
	je	.LBB15_59
.LBB15_50:                              # =>This Inner Loop Header: Depth=1
	cmpb	$0, 144(%rsp)                   # 1-byte Folded Reload
	je	.LBB15_55
# %bb.51:                               #   in Loop: Header=BB15_50 Depth=1
.Ltmp330:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movq	%rbp, %rdi
	movl	$1, %esi
	movq	%r14, %rdx
	movl	$1, %ecx
	xorl	%r8d, %r8d
	xorl	%r9d, %r9d
	callq	__hipPushCallConfiguration@PLT
.Ltmp331:                               # EH_LABEL
# %bb.52:                               #   in Loop: Header=BB15_50 Depth=1
	testl	%eax, %eax
	jne	.LBB15_49
# %bb.53:                               #   in Loop: Header=BB15_50 Depth=1
	movq	48(%rsp), %rax
	movq	160(%rsp), %rcx
	movq	168(%rsp), %rdx
	movq	%rax, 128(%rsp)
	movq	%rcx, 120(%rsp)
	movq	%rdx, 112(%rsp)
	movq	24(%rsp), %rax                  # 8-byte Reload
	movl	%eax, 12(%rsp)
	movq	32(%rsp), %rax                  # 8-byte Reload
	movl	%eax, 148(%rsp)
	leaq	128(%rsp), %rax
	movq	%rax, 176(%rsp)
	leaq	120(%rsp), %rax
	movq	%rax, 184(%rsp)
	leaq	112(%rsp), %rax
	movq	%rax, 192(%rsp)
	leaq	12(%rsp), %rax
	movq	%rax, 200(%rsp)
	leaq	148(%rsp), %rax
	movq	%rax, 208(%rsp)
.Ltmp332:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	leaq	96(%rsp), %rdi
	leaq	80(%rsp), %rsi
	leaq	72(%rsp), %rdx
	leaq	64(%rsp), %rcx
	callq	__hipPopCallConfiguration@PLT
.Ltmp333:                               # EH_LABEL
# %bb.54:                               # %.noexc136
                                        #   in Loop: Header=BB15_50 Depth=1
	movq	96(%rsp), %rsi
	movl	104(%rsp), %edx
	movq	80(%rsp), %rcx
	movl	88(%rsp), %r8d
.Ltmp334:                               # EH_LABEL
	.cfi_escape 0x2e, 0x10
	movq	_Z20decode_only_repackedPKhS0_Phii@GOTPCREL(%rip), %rdi
	leaq	176(%rsp), %r9
	pushq	64(%rsp)
	.cfi_adjust_cfa_offset 8
	pushq	80(%rsp)
	.cfi_adjust_cfa_offset 8
	callq	hipLaunchKernel@PLT
	addq	$16, %rsp
	.cfi_adjust_cfa_offset -16
.Ltmp335:                               # EH_LABEL
	jmp	.LBB15_49
	.p2align	4
.LBB15_55:                              #   in Loop: Header=BB15_50 Depth=1
.Ltmp324:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movq	%rbp, %rdi
	movl	$1, %esi
	movq	%r14, %rdx
	movl	$1, %ecx
	xorl	%r8d, %r8d
	xorl	%r9d, %r9d
	callq	__hipPushCallConfiguration@PLT
.Ltmp325:                               # EH_LABEL
# %bb.56:                               #   in Loop: Header=BB15_50 Depth=1
	testl	%eax, %eax
	jne	.LBB15_49
# %bb.57:                               #   in Loop: Header=BB15_50 Depth=1
	movq	48(%rsp), %rax
	movq	160(%rsp), %rcx
	movq	168(%rsp), %rdx
	movq	%rax, 128(%rsp)
	movq	%rcx, 120(%rsp)
	movq	%rdx, 112(%rsp)
	movq	24(%rsp), %rax                  # 8-byte Reload
	movl	%eax, 12(%rsp)
	movq	32(%rsp), %rax                  # 8-byte Reload
	movl	%eax, 148(%rsp)
	movl	140(%rsp), %eax                 # 4-byte Reload
	movl	%eax, 260(%rsp)
	leaq	128(%rsp), %rax
	movq	%rax, 176(%rsp)
	leaq	120(%rsp), %rax
	movq	%rax, 184(%rsp)
	leaq	112(%rsp), %rax
	movq	%rax, 192(%rsp)
	leaq	12(%rsp), %rax
	movq	%rax, 200(%rsp)
	leaq	148(%rsp), %rax
	movq	%rax, 208(%rsp)
	leaq	260(%rsp), %rax
	movq	%rax, 216(%rsp)
.Ltmp326:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	leaq	96(%rsp), %rdi
	leaq	80(%rsp), %rsi
	leaq	72(%rsp), %rdx
	leaq	64(%rsp), %rcx
	callq	__hipPopCallConfiguration@PLT
.Ltmp327:                               # EH_LABEL
# %bb.58:                               # %.noexc145
                                        #   in Loop: Header=BB15_50 Depth=1
	movq	96(%rsp), %rsi
	movl	104(%rsp), %edx
	movq	80(%rsp), %rcx
	movl	88(%rsp), %r8d
.Ltmp328:                               # EH_LABEL
	.cfi_escape 0x2e, 0x10
	movq	_Z21decode_only_scatteredPKhS0_Phiii@GOTPCREL(%rip), %rdi
	leaq	176(%rsp), %r9
	pushq	64(%rsp)
	.cfi_adjust_cfa_offset 8
	pushq	80(%rsp)
	.cfi_adjust_cfa_offset 8
	callq	hipLaunchKernel@PLT
	addq	$16, %rsp
	.cfi_adjust_cfa_offset -16
.Ltmp329:                               # EH_LABEL
	jmp	.LBB15_49
.LBB15_59:                              # %._crit_edge224
	movq	40(%rsp), %rdi
.Ltmp337:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	xorl	%esi, %esi
	callq	hipEventRecord@PLT
.Ltmp338:                               # EH_LABEL
# %bb.60:
	movq	40(%rsp), %rdi
.Ltmp339:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	callq	hipEventSynchronize@PLT
.Ltmp340:                               # EH_LABEL
# %bb.61:
	movq	56(%rsp), %rsi
	movq	40(%rsp), %rdx
.Ltmp342:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	leaq	176(%rsp), %rdi
	callq	hipEventElapsedTime@PLT
.Ltmp343:                               # EH_LABEL
# %bb.62:
	cvtsi2ss	%ebx, %xmm0
	movss	176(%rsp), %xmm1                # xmm1 = mem[0],zero,zero,zero
	divss	%xmm0, %xmm1
	movss	%xmm1, 176(%rsp)
	movq	48(%rsp), %rdi
.Ltmp344:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	callq	hipFree@PLT
.Ltmp345:                               # EH_LABEL
# %bb.63:
	movq	160(%rsp), %rdi
.Ltmp346:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	callq	hipFree@PLT
.Ltmp347:                               # EH_LABEL
# %bb.64:
	movq	168(%rsp), %rdi
.Ltmp348:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	callq	hipFree@PLT
.Ltmp349:                               # EH_LABEL
# %bb.65:
	movq	56(%rsp), %rdi
.Ltmp350:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	callq	hipEventDestroy@PLT
.Ltmp351:                               # EH_LABEL
# %bb.66:
	movq	40(%rsp), %rdi
.Ltmp352:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	callq	hipEventDestroy@PLT
.Ltmp353:                               # EH_LABEL
# %bb.67:
	movss	176(%rsp), %xmm0                # xmm0 = mem[0],zero,zero,zero
	movss	%xmm0, 248(%rsp)                # 4-byte Spill
	movq	16(%rsp), %rdi                  # 8-byte Reload
	testq	%rdi, %rdi
	movq	224(%rsp), %rbx                 # 8-byte Reload
	je	.LBB15_69
# %bb.68:
	movq	152(%rsp), %rsi                 # 8-byte Reload
	subq	%rdi, %rsi
	.cfi_escape 0x2e, 0x00
	callq	_ZdlPvm@PLT
.LBB15_69:                              # %_ZNSt6vectorIhSaIhEED2Ev.exit149
	.cfi_escape 0x2e, 0x00
	movl	$2048, %esi                     # imm = 0x800
	movq	%rbx, %rdi
	callq	_ZdlPvm@PLT
	testq	%r12, %r12
	je	.LBB15_71
# %bb.70:
	subq	%r12, %r13
	.cfi_escape 0x2e, 0x00
	movq	%r12, %rdi
	movq	%r13, %rsi
	callq	_ZdlPvm@PLT
.LBB15_71:                              # %_ZNSt6vectorIhSaIhEED2Ev.exit151
	movss	248(%rsp), %xmm0                # 4-byte Reload
                                        # xmm0 = mem[0],zero,zero,zero
	cvtss2sd	%xmm0, %xmm0
	addq	$5368, %rsp                     # imm = 0x14F8
	.cfi_def_cfa_offset 56
	popq	%rbx
	.cfi_def_cfa_offset 48
	popq	%r12
	.cfi_def_cfa_offset 40
	popq	%r13
	.cfi_def_cfa_offset 32
	popq	%r14
	.cfi_def_cfa_offset 24
	popq	%r15
	.cfi_def_cfa_offset 16
	popq	%rbp
	.cfi_def_cfa_offset 8
	retq
.LBB15_72:                              # %.noexc
	.cfi_def_cfa_offset 5424
	.cfi_escape 0x2e, 0x00
	leaq	.L.str.55(%rip), %rdi
	callq	_ZSt20__throw_length_errorPKc@PLT
.LBB15_73:
	movq	$0, 16(%rsp)                    # 8-byte Folded Spill
.Ltmp296:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	leaq	.L.str.56(%rip), %rdi
	movq	$0, 152(%rsp)                   # 8-byte Folded Spill
	callq	_ZSt20__throw_length_errorPKc@PLT
.Ltmp297:                               # EH_LABEL
# %bb.74:                               # %.noexc158
.LBB15_75:
.Ltmp279:                               # EH_LABEL
	movq	%rax, %r14
	jmp	.LBB15_89
.LBB15_76:
.Ltmp302:                               # EH_LABEL
	jmp	.LBB15_85
.LBB15_77:                              # %.thread
.Ltmp293:                               # EH_LABEL
	movq	%rax, %r14
	jmp	.LBB15_88
.LBB15_78:
.Ltmp341:                               # EH_LABEL
	jmp	.LBB15_83
.LBB15_79:
.Ltmp354:                               # EH_LABEL
	jmp	.LBB15_83
.LBB15_80:                              # %.thread239
.Ltmp276:                               # EH_LABEL
	movq	%rax, %r14
	jmp	.LBB15_90
.LBB15_81:
.Ltmp282:                               # EH_LABEL
	movq	%rax, %r14
	jmp	.LBB15_88
.LBB15_82:
.Ltmp336:                               # EH_LABEL
.LBB15_83:
	movq	%rax, %r14
	movq	232(%rsp), %r12                 # 8-byte Reload
	jmp	.LBB15_86
.LBB15_84:
.Ltmp317:                               # EH_LABEL
.LBB15_85:
	movq	%rax, %r14
.LBB15_86:
	movq	224(%rsp), %r15                 # 8-byte Reload
	movq	16(%rsp), %rdi                  # 8-byte Reload
	testq	%rdi, %rdi
	je	.LBB15_88
# %bb.87:
	movq	152(%rsp), %rsi                 # 8-byte Reload
	subq	%rdi, %rsi
	.cfi_escape 0x2e, 0x00
	callq	_ZdlPvm@PLT
.LBB15_88:                              # %_ZNSt6vectorIhSaIhEED2Ev.exit155
	.cfi_escape 0x2e, 0x00
	movl	$2048, %esi                     # imm = 0x800
	movq	%r15, %rdi
	callq	_ZdlPvm@PLT
.LBB15_89:
	testq	%r12, %r12
	je	.LBB15_91
.LBB15_90:
	subq	%r12, %r13
	.cfi_escape 0x2e, 0x00
	movq	%r12, %rdi
	movq	%r13, %rsi
	callq	_ZdlPvm@PLT
.LBB15_91:                              # %_ZNSt6vectorIhSaIhEED2Ev.exit157
	.cfi_escape 0x2e, 0x00
	movq	%r14, %rdi
	callq	_Unwind_Resume@PLT
.Lfunc_end15:
	.size	_Z21bench_decode_isolatediibi, .Lfunc_end15-_Z21bench_decode_isolatediibi
	.cfi_endproc
	.section	.gcc_except_table,"a",@progbits
	.p2align	2, 0x0
GCC_except_table15:
.Lexception3:
	.byte	255                             # @LPStart Encoding = omit
	.byte	255                             # @TType Encoding = omit
	.byte	1                               # Call site Encoding = uleb128
	.uleb128 .Lcst_end3-.Lcst_begin3
.Lcst_begin3:
	.uleb128 .Lfunc_begin3-.Lfunc_begin3    # >> Call Site 1 <<
	.uleb128 .Ltmp274-.Lfunc_begin3         #   Call between .Lfunc_begin3 and .Ltmp274
	.byte	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp274-.Lfunc_begin3         # >> Call Site 2 <<
	.uleb128 .Ltmp275-.Ltmp274              #   Call between .Ltmp274 and .Ltmp275
	.uleb128 .Ltmp276-.Lfunc_begin3         #     jumps to .Ltmp276
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp277-.Lfunc_begin3         # >> Call Site 3 <<
	.uleb128 .Ltmp278-.Ltmp277              #   Call between .Ltmp277 and .Ltmp278
	.uleb128 .Ltmp279-.Lfunc_begin3         #     jumps to .Ltmp279
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp278-.Lfunc_begin3         # >> Call Site 4 <<
	.uleb128 .Ltmp280-.Ltmp278              #   Call between .Ltmp278 and .Ltmp280
	.byte	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp280-.Lfunc_begin3         # >> Call Site 5 <<
	.uleb128 .Ltmp281-.Ltmp280              #   Call between .Ltmp280 and .Ltmp281
	.uleb128 .Ltmp282-.Lfunc_begin3         #     jumps to .Ltmp282
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp283-.Lfunc_begin3         # >> Call Site 6 <<
	.uleb128 .Ltmp288-.Ltmp283              #   Call between .Ltmp283 and .Ltmp288
	.uleb128 .Ltmp293-.Lfunc_begin3         #     jumps to .Ltmp293
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp294-.Lfunc_begin3         # >> Call Site 7 <<
	.uleb128 .Ltmp295-.Ltmp294              #   Call between .Ltmp294 and .Ltmp295
	.uleb128 .Ltmp302-.Lfunc_begin3         #     jumps to .Ltmp302
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp295-.Lfunc_begin3         # >> Call Site 8 <<
	.uleb128 .Ltmp289-.Ltmp295              #   Call between .Ltmp295 and .Ltmp289
	.byte	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp289-.Lfunc_begin3         # >> Call Site 9 <<
	.uleb128 .Ltmp292-.Ltmp289              #   Call between .Ltmp289 and .Ltmp292
	.uleb128 .Ltmp293-.Lfunc_begin3         #     jumps to .Ltmp293
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp298-.Lfunc_begin3         # >> Call Site 10 <<
	.uleb128 .Ltmp301-.Ltmp298              #   Call between .Ltmp298 and .Ltmp301
	.uleb128 .Ltmp302-.Lfunc_begin3         #     jumps to .Ltmp302
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp309-.Lfunc_begin3         # >> Call Site 11 <<
	.uleb128 .Ltmp316-.Ltmp309              #   Call between .Ltmp309 and .Ltmp316
	.uleb128 .Ltmp317-.Lfunc_begin3         #     jumps to .Ltmp317
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp318-.Lfunc_begin3         # >> Call Site 12 <<
	.uleb128 .Ltmp323-.Ltmp318              #   Call between .Ltmp318 and .Ltmp323
	.uleb128 .Ltmp341-.Lfunc_begin3         #     jumps to .Ltmp341
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp330-.Lfunc_begin3         # >> Call Site 13 <<
	.uleb128 .Ltmp329-.Ltmp330              #   Call between .Ltmp330 and .Ltmp329
	.uleb128 .Ltmp336-.Lfunc_begin3         #     jumps to .Ltmp336
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp337-.Lfunc_begin3         # >> Call Site 14 <<
	.uleb128 .Ltmp340-.Ltmp337              #   Call between .Ltmp337 and .Ltmp340
	.uleb128 .Ltmp341-.Lfunc_begin3         #     jumps to .Ltmp341
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp342-.Lfunc_begin3         # >> Call Site 15 <<
	.uleb128 .Ltmp353-.Ltmp342              #   Call between .Ltmp342 and .Ltmp353
	.uleb128 .Ltmp354-.Lfunc_begin3         #     jumps to .Ltmp354
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp353-.Lfunc_begin3         # >> Call Site 16 <<
	.uleb128 .Ltmp296-.Ltmp353              #   Call between .Ltmp353 and .Ltmp296
	.byte	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp296-.Lfunc_begin3         # >> Call Site 17 <<
	.uleb128 .Ltmp297-.Ltmp296              #   Call between .Ltmp296 and .Ltmp297
	.uleb128 .Ltmp302-.Lfunc_begin3         #     jumps to .Ltmp302
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp297-.Lfunc_begin3         # >> Call Site 18 <<
	.uleb128 .Lfunc_end15-.Ltmp297          #   Call between .Ltmp297 and .Lfunc_end15
	.byte	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
.Lcst_end3:
	.p2align	2, 0x0
                                        # -- End function
	.section	.rodata.cst4,"aM",@progbits,4
	.p2align	2, 0x0                          # -- Begin function _Z11validate_m6v
.LCPI16_0:
	.long	0x3f000000                      # float 0.5
.LCPI16_1:
	.long	0x41200000                      # float 10
	.section	.rodata.cst16,"aM",@progbits,16
	.p2align	4, 0x0
.LCPI16_2:
	.long	8                               # 0x8
	.long	8                               # 0x8
	.long	8                               # 0x8
	.long	8                               # 0x8
.LCPI16_3:
	.long	0                               # 0x0
	.long	512                             # 0x200
	.long	1024                            # 0x400
	.long	1536                            # 0x600
.LCPI16_4:
	.long	0x7fffffff                      # float NaN
	.long	0x7fffffff                      # float NaN
	.long	0x7fffffff                      # float NaN
	.long	0x7fffffff                      # float NaN
	.section	.rodata.cst8,"aM",@progbits,8
	.p2align	3, 0x0
.LCPI16_5:
	.quad	0x3f50624dd2f1a9fc              # double 0.001
	.text
	.globl	_Z11validate_m6v
	.prefalign	4, .Lfunc_end16, nop
	.type	_Z11validate_m6v,@function
_Z11validate_m6v:                       # @_Z11validate_m6v
.Lfunc_begin4:
	.cfi_startproc
	.cfi_personality 155, DW.ref.__gxx_personality_v0
	.cfi_lsda 27, .Lexception4
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	pushq	%r15
	.cfi_def_cfa_offset 24
	pushq	%r14
	.cfi_def_cfa_offset 32
	pushq	%r13
	.cfi_def_cfa_offset 40
	pushq	%r12
	.cfi_def_cfa_offset 48
	pushq	%rbx
	.cfi_def_cfa_offset 56
	subq	$5320, %rsp                     # imm = 0x14C8
	.cfi_def_cfa_offset 5376
	.cfi_offset %rbx, -56
	.cfi_offset %r12, -48
	.cfi_offset %r13, -40
	.cfi_offset %r14, -32
	.cfi_offset %r15, -24
	.cfi_offset %rbp, -16
	movq	$123, 320(%rsp)
	movl	$123, %ecx
	movl	$2, %eax
	.p2align	4
.LBB16_1:                               # =>This Inner Loop Header: Depth=1
	movq	%rcx, %rdx
	shrq	$30, %rdx
	xorq	%rcx, %rdx
	imulq	$1812433253, %rdx, %rcx         # imm = 0x6C078965
	addq	%rax, %rcx
	decq	%rcx
	movl	%ecx, %edx
	movq	%rdx, 312(%rsp,%rax,8)
	cmpq	$624, %rax                      # imm = 0x270
	je	.LBB16_3
# %bb.2:                                #   in Loop: Header=BB16_1 Depth=1
	shrl	$30, %edx
	xorl	%edx, %ecx
	imull	$1812433253, %ecx, %ecx         # imm = 0x6C078965
	addl	%eax, %ecx
	movq	%rcx, 320(%rsp,%rax,8)
	addq	$2, %rax
	jmp	.LBB16_1
.LBB16_3:                               # %_ZNSt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EEC2Em.exit
	movq	$624, 5312(%rsp)                # imm = 0x270
	.cfi_escape 0x2e, 0x00
	movl	$4096, %edi                     # imm = 0x1000
	callq	_Znwm@PLT
	movq	%rax, %r13
	.cfi_escape 0x2e, 0x00
	xorl	%ebx, %ebx
	movl	$4096, %edx                     # imm = 0x1000
	movq	%rax, %rdi
	xorl	%esi, %esi
	callq	memset@PLT
	leaq	320(%rsp), %r14
	.p2align	4
.LBB16_4:                               # =>This Inner Loop Header: Depth=1
.Ltmp355:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movq	%r14, %rdi
	callq	_ZNSt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EEclEv
.Ltmp356:                               # EH_LABEL
# %bb.5:                                #   in Loop: Header=BB16_4 Depth=1
	movb	%al, (%r13,%rbx)
	incq	%rbx
	cmpq	$4096, %rbx                     # imm = 0x1000
	jne	.LBB16_4
# %bb.6:
.Ltmp358:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movl	$2048, %edi                     # imm = 0x800
	callq	_Znwm@PLT
.Ltmp359:                               # EH_LABEL
# %bb.7:                                # %.lr.ph.preheader
	movq	%rax, %rbp
	.cfi_escape 0x2e, 0x00
	xorl	%ebx, %ebx
	movl	$2048, %edx                     # imm = 0x800
	movq	%rax, %rdi
	xorl	%esi, %esi
	callq	memset@PLT
	leaq	320(%rsp), %r15
	movabsq	$-3689348814741910323, %r14     # imm = 0xCCCCCCCCCCCCCCCD
	jmp	.LBB16_9
	.p2align	4
.LBB16_8:                               #   in Loop: Header=BB16_9 Depth=1
	incq	%rbx
	cmpq	$2048, %rbx                     # imm = 0x800
	je	.LBB16_12
.LBB16_9:                               # %.lr.ph
                                        # =>This Inner Loop Header: Depth=1
.Ltmp361:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movq	%r15, %rdi
	callq	_ZNSt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EEclEv
.Ltmp362:                               # EH_LABEL
# %bb.10:                               #   in Loop: Header=BB16_9 Depth=1
	movq	%rax, %rcx
	mulq	%r14
	shrq	$2, %rdx
	leal	(%rdx,%rdx,4), %eax
	subl	%eax, %ecx
	xorps	%xmm0, %xmm0
	cvtsi2ss	%ecx, %xmm0
	mulss	.LCPI16_0(%rip), %xmm0
	mulss	.LCPI16_1(%rip), %xmm0
	cvttss2si	%xmm0, %eax
	movb	%al, (%rbp,%rbx)
	orl	$128, %eax
	movzbl	%al, %eax
	cmpl	$255, %eax
	jne	.LBB16_8
# %bb.11:                               #   in Loop: Header=BB16_9 Depth=1
	movb	$0, (%rbp,%rbx)
	jmp	.LBB16_8
.LBB16_12:                              # %._crit_edge
.Ltmp364:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movl	$4096, %edi                     # imm = 0x1000
	callq	_Znwm@PLT
.Ltmp365:                               # EH_LABEL
# %bb.13:
	.cfi_escape 0x2e, 0x00
	xorl	%ebx, %ebx
	movl	$4096, %edx                     # imm = 0x1000
	movq	%rax, 72(%rsp)                  # 8-byte Spill
	movq	%rax, %rdi
	xorl	%esi, %esi
	callq	memset@PLT
	leaq	320(%rsp), %r15
	movabsq	$-2049638230412172401, %r14     # imm = 0xE38E38E38E38E38F
	leaq	208(%rsp), %r12
	.p2align	4
.LBB16_14:                              # =>This Inner Loop Header: Depth=1
.Ltmp367:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movq	%r15, %rdi
	callq	_ZNSt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EEclEv
.Ltmp368:                               # EH_LABEL
# %bb.15:                               #   in Loop: Header=BB16_14 Depth=1
	movq	%rax, %rcx
	mulq	%r14
	shrq	$3, %rdx
	leaq	(%rdx,%rdx,8), %rax
	negq	%rax
	addq	%rcx, %rax
	addq	$-4, %rax
	testq	%rax, %rax
	js	.LBB16_17
# %bb.16:                               #   in Loop: Header=BB16_14 Depth=1
	xorps	%xmm0, %xmm0
	cvtsi2ss	%rax, %xmm0
	jmp	.LBB16_18
	.p2align	4
.LBB16_17:                              #   in Loop: Header=BB16_14 Depth=1
	movq	%rax, %rcx
	shrq	%rcx
	andl	$1, %eax
	orq	%rcx, %rax
	xorps	%xmm0, %xmm0
	cvtsi2ss	%rax, %xmm0
	addss	%xmm0, %xmm0
.LBB16_18:                              #   in Loop: Header=BB16_14 Depth=1
	mulss	.LCPI16_0(%rip), %xmm0
.Ltmp370:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movq	%r12, %rdi
	callq	_ZN14__hip_fp8_e4m3C2Ef
.Ltmp371:                               # EH_LABEL
# %bb.19:                               #   in Loop: Header=BB16_14 Depth=1
	movzbl	208(%rsp), %eax
	movq	72(%rsp), %rcx                  # 8-byte Reload
	movb	%al, (%rcx,%rbx)
	incq	%rbx
	cmpq	$4096, %rbx                     # imm = 0x1000
	jne	.LBB16_14
# %bb.20:
.Ltmp373:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movl	$16, %edi
	callq	_Znwm@PLT
.Ltmp374:                               # EH_LABEL
# %bb.21:
	movq	%rax, %r12
	movaps	.LCPI16_2(%rip), %xmm0          # xmm0 = [8,8,8,8]
	movups	%xmm0, (%rax)
.Ltmp376:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movl	$16, %edi
	movq	%r13, 64(%rsp)                  # 8-byte Spill
	movq	%rbp, 176(%rsp)                 # 8-byte Spill
	movq	%rax, 296(%rsp)                 # 8-byte Spill
	callq	_Znwm@PLT
.Ltmp377:                               # EH_LABEL
# %bb.22:                               # %.lr.ph.i.i.i.i.i.preheader
	movq	%rax, %rbx
	movaps	.LCPI16_3(%rip), %xmm0          # xmm0 = [0,512,1024,1536]
	movups	%xmm0, (%rax)
.Ltmp379:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movl	$8192, %edi                     # imm = 0x2000
	movq	%rax, 288(%rsp)                 # 8-byte Spill
	callq	_Znwm@PLT
.Ltmp380:                               # EH_LABEL
# %bb.23:
	movq	%rax, %r14
	.cfi_escape 0x2e, 0x00
	xorl	%r13d, %r13d
	movl	$8192, %edx                     # imm = 0x2000
	movq	%rax, %rdi
	xorl	%esi, %esi
	callq	memset@PLT
	movzbl	(%r12), %ecx
	movzbl	4(%r12), %eax
	movl	$-1, %edx
	movl	$-1, %r8d
	shll	%cl, %r8d
	notl	%r8d
	movl	(%rbx), %ebp
	movl	4(%rbx), %esi
	movl	$-1, %r9d
	movl	%eax, %ecx
	shll	%cl, %r9d
	leal	1(%rbp), %eax
	movq	%rax, 16(%rsp)                  # 8-byte Spill
	notl	%r9d
	movzbl	8(%r12), %ecx
	movl	$-1, %r10d
	shll	%cl, %r10d
	movq	%rsi, 40(%rsp)                  # 8-byte Spill
	leal	1(%rsi), %eax
	movq	%rax, 304(%rsp)                 # 8-byte Spill
	notl	%r10d
	movl	8(%rbx), %eax
	movzbl	12(%r12), %ecx
	shll	%cl, %edx
	movq	%rax, 32(%rsp)                  # 8-byte Spill
	incl	%eax
	movq	%rax, 48(%rsp)                  # 8-byte Spill
	notl	%edx
	movl	12(%rbx), %ebx
	movq	64(%rsp), %rax                  # 8-byte Reload
	leaq	3(%rax), %rsi
	movq	%r14, 192(%rsp)                 # 8-byte Spill
	addq	$7, %r14
	movl	%ebx, %eax
	movq	%rax, 272(%rsp)                 # 8-byte Spill
	incl	%ebx
	movq	304(%rsp), %r15                 # 8-byte Reload
	.p2align	4
.LBB16_24:                              # %.preheader653
                                        # =>This Loop Header: Depth=1
                                        #     Child Loop BB16_25 Depth 2
	movq	%r13, 312(%rsp)                 # 8-byte Spill
	xorl	%edi, %edi
	movq	176(%rsp), %r13                 # 8-byte Reload
	.p2align	4
.LBB16_25:                              # %._crit_edge670
                                        #   Parent Loop BB16_24 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	movl	-3(%rsi,%rdi,4), %r11d
	movl	%r8d, %ecx
	andl	%r11d, %ecx
	leal	(%rbp,%rcx,2), %eax
	cltq
	movzbl	(%r13,%rax), %eax
	movb	%al, -7(%r14,%rdi,8)
	movq	16(%rsp), %rax                  # 8-byte Reload
	leal	(%rax,%rcx,2), %eax
	cltq
	movzbl	(%r13,%rax), %eax
	movb	%al, -6(%r14,%rdi,8)
	movzbl	(%r12), %ecx
	movq	%r11, %rax
	shrq	%cl, %rax
	andl	%r9d, %eax
	movq	40(%rsp), %rcx                  # 8-byte Reload
	leal	(%rcx,%rax,2), %eax
	cltq
	movzbl	(%r12), %ecx
	movq	%rbp, %r12
	movl	%r8d, %ebp
	movq	%r11, %r8
	shrq	%cl, %r8
	movzbl	(%r13,%rax), %eax
	movb	%al, -5(%r14,%rdi,8)
	andl	%r9d, %r8d
	leal	(%r15,%r8,2), %eax
	movl	%ebp, %r8d
	movq	%r12, %rbp
	movq	296(%rsp), %r12                 # 8-byte Reload
	cltq
	movzbl	(%r13,%rax), %eax
	movb	%al, -4(%r14,%rdi,8)
	movl	4(%r12), %ecx
	addl	(%r12), %ecx
	movq	%r11, %rax
                                        # kill: def $cl killed $cl killed $ecx
	shrq	%cl, %rax
	andl	%r10d, %eax
	movq	32(%rsp), %rcx                  # 8-byte Reload
	leal	(%rcx,%rax,2), %eax
	cltq
	movzbl	(%r13,%rax), %eax
	movb	%al, -3(%r14,%rdi,8)
	movl	4(%r12), %ecx
	addl	(%r12), %ecx
	movq	%r11, %rax
                                        # kill: def $cl killed $cl killed $ecx
	shrq	%cl, %rax
	andl	%r10d, %eax
	movq	48(%rsp), %rcx                  # 8-byte Reload
	leal	(%rcx,%rax,2), %eax
	cltq
	movzbl	(%r13,%rax), %eax
	movb	%al, -2(%r14,%rdi,8)
	movl	4(%r12), %ecx
	addl	(%r12), %ecx
	addl	8(%r12), %ecx
	movq	%r11, %rax
                                        # kill: def $cl killed $cl killed $ecx
	shrq	%cl, %rax
	andl	%edx, %eax
	movq	272(%rsp), %rcx                 # 8-byte Reload
	leal	(%rcx,%rax,2), %eax
	cltq
	movzbl	(%r13,%rax), %eax
	movl	4(%r12), %ecx
	addl	(%r12), %ecx
	addl	8(%r12), %ecx
                                        # kill: def $cl killed $cl killed $ecx
	shrq	%cl, %r11
	movb	%al, -1(%r14,%rdi,8)
	andl	%edx, %r11d
	leal	(%rbx,%r11,2), %eax
	cltq
	movzbl	(%r13,%rax), %eax
	movb	%al, (%r14,%rdi,8)
	incq	%rdi
	cmpq	$32, %rdi
	jne	.LBB16_25
# %bb.26:                               # %.loopexit
                                        #   in Loop: Header=BB16_24 Depth=1
	movq	312(%rsp), %r13                 # 8-byte Reload
	incq	%r13
	subq	$-128, %rsi
	addq	$256, %r14                      # imm = 0x100
	cmpq	$32, %r13
	jne	.LBB16_24
# %bb.27:
.Ltmp382:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movl	$16384, %edi                    # imm = 0x4000
	callq	_Znwm@PLT
.Ltmp383:                               # EH_LABEL
# %bb.28:
	.cfi_escape 0x2e, 0x00
	movl	$16384, %edx                    # imm = 0x4000
	movq	%rax, 40(%rsp)                  # 8-byte Spill
	movq	%rax, %rdi
	xorl	%esi, %esi
	callq	memset@PLT
.Ltmp385:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movl	$32768, %edi                    # imm = 0x8000
	movq	72(%rsp), %r14                  # 8-byte Reload
	callq	_Znwm@PLT
.Ltmp386:                               # EH_LABEL
# %bb.29:
	.cfi_escape 0x2e, 0x00
	xorl	%ebx, %ebx
	movl	$32768, %edx                    # imm = 0x8000
	movq	%rax, 32(%rsp)                  # 8-byte Spill
	movq	%rax, %rdi
	xorl	%esi, %esi
	callq	memset@PLT
	jmp	.LBB16_32
.LBB16_30:                              #   in Loop: Header=BB16_32 Depth=1
	shll	$20, %edx
	andl	$-128, %eax
	shll	$24, %eax
	orl	%edx, %eax
	shll	$23, %ecx
	addl	$1006632960, %ecx               # imm = 0x3C000000
	orl	%eax, %ecx
	movl	%ecx, %edx
	.p2align	4
.LBB16_31:                              # %_ZNK14__hip_fp8_e4m3cvfEv.exit
                                        #   in Loop: Header=BB16_32 Depth=1
	movq	40(%rsp), %rax                  # 8-byte Reload
	movl	%edx, (%rax,%rbx,4)
	incq	%rbx
	cmpq	$4096, %rbx                     # imm = 0x1000
	je	.LBB16_37
.LBB16_32:                              # =>This Inner Loop Header: Depth=1
	movzbl	(%r14,%rbx), %eax
	movl	$0, %edx
	testl	%eax, %eax
	je	.LBB16_31
# %bb.33:                               #   in Loop: Header=BB16_32 Depth=1
	movl	$-2147483648, %edx              # imm = 0x80000000
	cmpq	$128, %rax
	je	.LBB16_31
# %bb.34:                               #   in Loop: Header=BB16_32 Depth=1
	movl	%eax, %ecx
	andl	$127, %ecx
	movl	$2139095041, %edx               # imm = 0x7F800001
	cmpl	$127, %ecx
	je	.LBB16_31
# %bb.35:                               #   in Loop: Header=BB16_32 Depth=1
	movl	%eax, %edx
	andl	$7, %edx
	shrl	$3, %ecx
	jne	.LBB16_30
# %bb.36:                               #   in Loop: Header=BB16_32 Depth=1
	bsrl	%edx, %esi
	xorl	$31, %esi
	leal	-28(%rsi), %ecx
                                        # kill: def $cl killed $cl killed $ecx
	shlq	%cl, %rdx
	movl	$29, %ecx
	subl	%esi, %ecx
	andl	$7, %edx
	jmp	.LBB16_30
.LBB16_37:                              # %.preheader651.preheader
	xorl	%eax, %eax
	movq	192(%rsp), %r8                  # 8-byte Reload
	movq	32(%rsp), %r9                   # 8-byte Reload
	jmp	.LBB16_40
.LBB16_38:                              #   in Loop: Header=BB16_40 Depth=1
	shll	$20, %esi
	andl	$-128, %edx
	shll	$24, %edx
	orl	%esi, %edx
	shll	$23, %ecx
	addl	$1006632960, %ecx               # imm = 0x3C000000
	orl	%edx, %ecx
	movl	%ecx, %esi
	.p2align	4
.LBB16_39:                              # %_ZNK14__hip_fp8_e4m3cvfEv.exit357
                                        #   in Loop: Header=BB16_40 Depth=1
	movl	%esi, (%r9,%rax,4)
	incq	%rax
	cmpq	$8192, %rax                     # imm = 0x2000
	je	.LBB16_45
.LBB16_40:                              # %.preheader651
                                        # =>This Inner Loop Header: Depth=1
	movzbl	(%r8,%rax), %edx
	movl	$0, %esi
	testl	%edx, %edx
	je	.LBB16_39
# %bb.41:                               #   in Loop: Header=BB16_40 Depth=1
	movl	$-2147483648, %esi              # imm = 0x80000000
	cmpq	$128, %rdx
	je	.LBB16_39
# %bb.42:                               #   in Loop: Header=BB16_40 Depth=1
	movl	%edx, %ecx
	andl	$127, %ecx
	movl	$2139095041, %esi               # imm = 0x7F800001
	cmpl	$127, %ecx
	je	.LBB16_39
# %bb.43:                               #   in Loop: Header=BB16_40 Depth=1
	movl	%edx, %esi
	andl	$7, %esi
	shrl	$3, %ecx
	jne	.LBB16_38
# %bb.44:                               #   in Loop: Header=BB16_40 Depth=1
	bsrl	%esi, %edi
	xorl	$31, %edi
	leal	-28(%rdi), %ecx
                                        # kill: def $cl killed $cl killed $ecx
	shlq	%cl, %rsi
	movl	$29, %ecx
	subl	%edi, %ecx
	andl	$7, %esi
	jmp	.LBB16_38
.LBB16_45:
.Ltmp388:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movl	$2048, %edi                     # imm = 0x800
	callq	_Znwm@PLT
.Ltmp389:                               # EH_LABEL
# %bb.46:                               # %.lr.ph.i.i.i.i.i348.preheader
	.cfi_escape 0x2e, 0x00
	xorl	%ebx, %ebx
	movl	$2048, %edx                     # imm = 0x800
	movq	%rax, %r14
	movq	%rax, %rdi
	xorl	%esi, %esi
	callq	memset@PLT
	movq	40(%rsp), %rax                  # 8-byte Reload
	addq	$12, %rax
	movq	32(%rsp), %rcx                  # 8-byte Reload
	addq	$12, %rcx
	.p2align	4
.LBB16_47:                              # %.preheader650
                                        # =>This Loop Header: Depth=1
                                        #     Child Loop BB16_48 Depth 2
                                        #       Child Loop BB16_49 Depth 3
	movq	%rbx, %rdx
	shlq	$7, %rdx
	addq	%r14, %rdx
	movq	%rcx, %rsi
	xorl	%edi, %edi
	.p2align	4
.LBB16_48:                              # %.preheader649
                                        #   Parent Loop BB16_47 Depth=1
                                        # =>  This Loop Header: Depth=2
                                        #       Child Loop BB16_49 Depth 3
	xorps	%xmm0, %xmm0
	xorl	%r8d, %r8d
	.p2align	4
.LBB16_49:                              #   Parent Loop BB16_47 Depth=1
                                        #     Parent Loop BB16_48 Depth=2
                                        # =>    This Inner Loop Header: Depth=3
	movss	-12(%rax,%r8,4), %xmm1          # xmm1 = mem[0],zero,zero,zero
	movss	-8(%rax,%r8,4), %xmm2           # xmm2 = mem[0],zero,zero,zero
	mulss	-12(%rsi,%r8,4), %xmm1
	mulss	-8(%rsi,%r8,4), %xmm2
	addss	%xmm1, %xmm0
	movss	-4(%rax,%r8,4), %xmm1           # xmm1 = mem[0],zero,zero,zero
	mulss	-4(%rsi,%r8,4), %xmm1
	addss	%xmm0, %xmm2
	movss	(%rax,%r8,4), %xmm0             # xmm0 = mem[0],zero,zero,zero
	mulss	(%rsi,%r8,4), %xmm0
	addss	%xmm2, %xmm1
	addss	%xmm1, %xmm0
	addq	$4, %r8
	cmpq	$256, %r8                       # imm = 0x100
	jne	.LBB16_49
# %bb.50:                               #   in Loop: Header=BB16_48 Depth=2
	movss	%xmm0, (%rdx,%rdi,4)
	incq	%rdi
	addq	$1024, %rsi                     # imm = 0x400
	cmpq	$32, %rdi
	jne	.LBB16_48
# %bb.51:                               # %_ZNSt6vectorIfSaIfEEC2EmRKfRKS0_.exit
                                        #   in Loop: Header=BB16_47 Depth=1
	incq	%rbx
	addq	$1024, %rax                     # imm = 0x400
	cmpq	$16, %rbx
	jne	.LBB16_47
# %bb.52:
.Ltmp391:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movl	$4096, %edi                     # imm = 0x1000
	callq	_Znwm@PLT
.Ltmp392:                               # EH_LABEL
# %bb.53:                               # %.preheader.i
	movq	%rax, %rbp
	.cfi_escape 0x2e, 0x00
	xorl	%ebx, %ebx
	movl	$4096, %edx                     # imm = 0x1000
	movq	%rax, %rdi
	xorl	%esi, %esi
	callq	memset@PLT
	movl	$240, %eax
	movq	64(%rsp), %rdi                  # 8-byte Reload
	.p2align	4
.LBB16_54:                              # =>This Inner Loop Header: Depth=1
	movq	%rbx, %rcx
	andq	$-128, %rcx
	addq	%rdi, %rcx
	movups	(%rbx,%rcx), %xmm0
	movups	%xmm0, -240(%rbp,%rax)
	movups	128(%rbx,%rcx), %xmm0
	movups	%xmm0, -224(%rbp,%rax)
	movups	256(%rbx,%rcx), %xmm0
	movups	%xmm0, -208(%rbp,%rax)
	movups	384(%rbx,%rcx), %xmm0
	movups	%xmm0, -192(%rbp,%rax)
	movups	512(%rbx,%rcx), %xmm0
	movups	%xmm0, -176(%rbp,%rax)
	movups	640(%rbx,%rcx), %xmm0
	movups	%xmm0, -160(%rbp,%rax)
	movups	768(%rbx,%rcx), %xmm0
	movups	%xmm0, -144(%rbp,%rax)
	movups	896(%rbx,%rcx), %xmm0
	movups	%xmm0, -128(%rbp,%rax)
	movups	1024(%rbx,%rcx), %xmm0
	movups	%xmm0, -112(%rbp,%rax)
	movups	1152(%rbx,%rcx), %xmm0
	movups	%xmm0, -96(%rbp,%rax)
	movups	1280(%rbx,%rcx), %xmm0
	movups	%xmm0, -80(%rbp,%rax)
	movups	1408(%rbx,%rcx), %xmm0
	movups	%xmm0, -64(%rbp,%rax)
	movups	1536(%rbx,%rcx), %xmm0
	movups	%xmm0, -48(%rbp,%rax)
	movups	1664(%rbx,%rcx), %xmm0
	movups	%xmm0, -32(%rbp,%rax)
	movups	1792(%rbx,%rcx), %xmm0
	movups	%xmm0, -16(%rbp,%rax)
	movups	1920(%rbx,%rcx), %xmm0
	movups	%xmm0, (%rbp,%rax)
	addq	$16, %rbx
	addq	$512, %rax                      # imm = 0x200
	cmpq	$128, %rbx
	jne	.LBB16_54
# %bb.55:                               # %._crit_edge.i
	movl	$496, %eax                      # imm = 0x1F0
	xorl	%ecx, %ecx
	.p2align	4
.LBB16_56:                              # =>This Inner Loop Header: Depth=1
	movq	%rcx, %rdx
	andq	$-128, %rdx
	leaq	(%rdi,%rcx), %rsi
	movups	2048(%rdx,%rsi), %xmm0
	movups	%xmm0, -240(%rbp,%rax)
	movups	2176(%rdx,%rsi), %xmm0
	movups	%xmm0, -224(%rbp,%rax)
	movups	2304(%rdx,%rsi), %xmm0
	movups	%xmm0, -208(%rbp,%rax)
	movups	2432(%rdx,%rsi), %xmm0
	movups	%xmm0, -192(%rbp,%rax)
	movups	2560(%rdx,%rsi), %xmm0
	movups	%xmm0, -176(%rbp,%rax)
	movups	2688(%rdx,%rsi), %xmm0
	movups	%xmm0, -160(%rbp,%rax)
	movups	2816(%rdx,%rsi), %xmm0
	movups	%xmm0, -144(%rbp,%rax)
	movups	2944(%rdx,%rsi), %xmm0
	movups	%xmm0, -128(%rbp,%rax)
	movups	3072(%rdx,%rsi), %xmm0
	movups	%xmm0, -112(%rbp,%rax)
	movups	3200(%rdx,%rsi), %xmm0
	movups	%xmm0, -96(%rbp,%rax)
	movups	3328(%rdx,%rsi), %xmm0
	movups	%xmm0, -80(%rbp,%rax)
	movups	3456(%rdx,%rsi), %xmm0
	movups	%xmm0, -64(%rbp,%rax)
	movups	3584(%rdx,%rsi), %xmm0
	movups	%xmm0, -48(%rbp,%rax)
	movups	3712(%rdx,%rsi), %xmm0
	movups	%xmm0, -32(%rbp,%rax)
	movups	3840(%rdx,%rsi), %xmm0
	movups	%xmm0, -16(%rbp,%rax)
	movups	3968(%rdx,%rsi), %xmm0
	movups	%xmm0, (%rbp,%rax)
	addq	$16, %rcx
	addq	$512, %rax                      # imm = 0x200
	cmpq	$128, %rcx
	jne	.LBB16_56
# %bb.57:                               # %._crit_edge.i.1
.Ltmp394:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	leaq	88(%rsp), %rdi
	movl	$4096, %esi                     # imm = 0x1000
	callq	hipMalloc@PLT
.Ltmp395:                               # EH_LABEL
# %bb.58:                               # %_ZL9hipMallocI14__hip_fp8_e4m3E10hipError_tPPT_m.exit
.Ltmp396:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	leaq	184(%rsp), %rdi
	movl	$4096, %esi                     # imm = 0x1000
	callq	hipMalloc@PLT
.Ltmp397:                               # EH_LABEL
# %bb.59:                               # %_ZL9hipMallocIhE10hipError_tPPT_m.exit
.Ltmp398:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	leaq	80(%rsp), %rdi
	movl	$2048, %esi                     # imm = 0x800
	callq	hipMalloc@PLT
.Ltmp399:                               # EH_LABEL
# %bb.60:                               # %_ZL9hipMallocIhE10hipError_tPPT_m.exit364
.Ltmp400:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	leaq	8(%rsp), %rdi
	movl	$2048, %esi                     # imm = 0x800
	callq	hipMalloc@PLT
.Ltmp401:                               # EH_LABEL
# %bb.61:                               # %_ZL9hipMallocIfE10hipError_tPPT_m.exit
	movq	88(%rsp), %rdi
.Ltmp402:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movl	$4096, %edx                     # imm = 0x1000
	movq	72(%rsp), %rsi                  # 8-byte Reload
	movl	$1, %ecx
	callq	hipMemcpy@PLT
.Ltmp403:                               # EH_LABEL
# %bb.62:
	movq	184(%rsp), %rdi
.Ltmp404:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movl	$4096, %edx                     # imm = 0x1000
	movq	%rbp, %rsi
	movl	$1, %ecx
	callq	hipMemcpy@PLT
.Ltmp405:                               # EH_LABEL
# %bb.63:
	movq	80(%rsp), %rdi
.Ltmp406:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movl	$2048, %edx                     # imm = 0x800
	movq	176(%rsp), %rsi                 # 8-byte Reload
	movl	$1, %ecx
	callq	hipMemcpy@PLT
.Ltmp407:                               # EH_LABEL
# %bb.64:
	movq	8(%rsp), %rdi
.Ltmp408:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movl	$2048, %edx                     # imm = 0x800
	xorl	%esi, %esi
	callq	hipMemset@PLT
.Ltmp409:                               # EH_LABEL
# %bb.65:
.Ltmp411:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movabsq	$4294967298, %rdi               # imm = 0x100000002
	movabsq	$4294967328, %rdx               # imm = 0x100000020
	movl	$1, %esi
	movl	$1, %ecx
	xorl	%r8d, %r8d
	xorl	%r9d, %r9d
	callq	__hipPushCallConfiguration@PLT
.Ltmp412:                               # EH_LABEL
# %bb.66:
	testl	%eax, %eax
	jne	.LBB16_69
# %bb.67:
	movq	88(%rsp), %rax
	movq	184(%rsp), %rcx
	movq	80(%rsp), %rdx
	movq	8(%rsp), %rsi
	movq	%rax, 168(%rsp)
	movq	%rcx, 160(%rsp)
	movq	%rdx, 152(%rsp)
	movq	%rsi, 144(%rsp)
	movl	$16, 24(%rsp)
	movl	$32, 4(%rsp)
	movl	$256, (%rsp)                    # imm = 0x100
	leaq	168(%rsp), %rax
	movq	%rax, 208(%rsp)
	leaq	160(%rsp), %rax
	movq	%rax, 216(%rsp)
	leaq	152(%rsp), %rax
	movq	%rax, 224(%rsp)
	leaq	144(%rsp), %rax
	movq	%rax, 232(%rsp)
	leaq	24(%rsp), %rax
	movq	%rax, 240(%rsp)
	leaq	4(%rsp), %rax
	movq	%rax, 248(%rsp)
	movq	%rsp, %rax
	movq	%rax, 256(%rsp)
.Ltmp413:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	leaq	128(%rsp), %rdi
	leaq	112(%rsp), %rsi
	leaq	104(%rsp), %rdx
	leaq	96(%rsp), %rcx
	callq	__hipPopCallConfiguration@PLT
.Ltmp414:                               # EH_LABEL
# %bb.68:                               # %.noexc366
	movq	128(%rsp), %rsi
	movl	136(%rsp), %edx
	movq	112(%rsp), %rcx
	movl	120(%rsp), %r8d
.Ltmp415:                               # EH_LABEL
	.cfi_escape 0x2e, 0x10
	movq	_Z21fused_m6_repacked_k32PK14__hip_fp8_e4m3PKhS3_Pfiii@GOTPCREL(%rip), %rdi
	leaq	208(%rsp), %r9
	pushq	96(%rsp)
	.cfi_adjust_cfa_offset 8
	pushq	112(%rsp)
	.cfi_adjust_cfa_offset 8
	callq	hipLaunchKernel@PLT
	addq	$16, %rsp
	.cfi_adjust_cfa_offset -16
.Ltmp416:                               # EH_LABEL
.LBB16_69:
.Ltmp417:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	callq	hipDeviceSynchronize@PLT
.Ltmp418:                               # EH_LABEL
# %bb.70:
.Ltmp420:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movl	$2048, %edi                     # imm = 0x800
	callq	_Znwm@PLT
.Ltmp421:                               # EH_LABEL
# %bb.71:
	.cfi_escape 0x2e, 0x00
	xorl	%r15d, %r15d
	movl	$2048, %edx                     # imm = 0x800
	movq	%rax, %rdi
	xorl	%esi, %esi
	movq	%rax, %r13
	callq	memset@PLT
	movq	8(%rsp), %rsi
.Ltmp423:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movl	$2048, %edx                     # imm = 0x800
	movq	%r13, 16(%rsp)                  # 8-byte Spill
	movq	%r13, %rdi
	movl	$2, %ecx
	callq	hipMemcpy@PLT
.Ltmp424:                               # EH_LABEL
# %bb.72:                               # %.preheader648.preheader
	xorps	%xmm2, %xmm2
	xorl	%eax, %eax
	movq	%r14, %rdx
	movq	16(%rsp), %rsi                  # 8-byte Reload
	.p2align	4
.LBB16_73:                              # %.preheader648
                                        # =>This Inner Loop Header: Depth=1
	movaps	%xmm2, %xmm0
	movss	(%rsi,%rax,4), %xmm2            # xmm2 = mem[0],zero,zero,zero
	subss	(%rdx,%rax,4), %xmm2
	andps	.LCPI16_4(%rip), %xmm2
	xorps	%xmm1, %xmm1
	cvtss2sd	%xmm2, %xmm1
	xorl	%ecx, %ecx
	ucomisd	.LCPI16_5(%rip), %xmm1
	seta	%cl
	maxss	%xmm0, %xmm2
	addl	%ecx, %r15d
	incq	%rax
	cmpq	$512, %rax                      # imm = 0x200
	jne	.LBB16_73
# %bb.74:
.Ltmp426:                               # EH_LABEL
	movaps	%xmm2, 48(%rsp)                 # 16-byte Spill
	.cfi_escape 0x2e, 0x00
	movq	_ZSt4cout@GOTPCREL(%rip), %rdi
	leaq	.L.str(%rip), %rsi
	movl	$26, %edx
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
.Ltmp427:                               # EH_LABEL
# %bb.75:                               # %_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc.exit
.Ltmp428:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movq	_ZSt4cout@GOTPCREL(%rip), %rdi
	movl	$16, %esi
	callq	_ZNSolsEi@PLT
.Ltmp429:                               # EH_LABEL
# %bb.76:
.Ltmp430:                               # EH_LABEL
	movq	%rax, %r13
	.cfi_escape 0x2e, 0x00
	leaq	.L.str.1(%rip), %rsi
	movl	$2, %edx
	movq	%rax, %rdi
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
.Ltmp431:                               # EH_LABEL
# %bb.77:                               # %_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc.exit373
.Ltmp432:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movq	%r13, %rdi
	movl	$32, %esi
	callq	_ZNSolsEi@PLT
.Ltmp433:                               # EH_LABEL
# %bb.78:
.Ltmp434:                               # EH_LABEL
	movq	%rax, %r13
	.cfi_escape 0x2e, 0x00
	leaq	.L.str.2(%rip), %rsi
	movl	$2, %edx
	movq	%rax, %rdi
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
.Ltmp435:                               # EH_LABEL
# %bb.79:                               # %_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc.exit375
.Ltmp436:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movq	%r13, %rdi
	movl	$256, %esi                      # imm = 0x100
	callq	_ZNSolsEi@PLT
.Ltmp437:                               # EH_LABEL
# %bb.80:
.Ltmp438:                               # EH_LABEL
	movq	%rax, %r13
	.cfi_escape 0x2e, 0x00
	leaq	.L.str.3(%rip), %rsi
	movl	$9, %edx
	movq	%rax, %rdi
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
.Ltmp439:                               # EH_LABEL
# %bb.81:                               # %_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc.exit377
	movaps	48(%rsp), %xmm0                 # 16-byte Reload
	cvtss2sd	%xmm0, %xmm0
.Ltmp440:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movq	%r13, %rdi
	movsd	%xmm0, 48(%rsp)                 # 8-byte Spill
	callq	_ZNSo9_M_insertIdEERSoT_@PLT
.Ltmp441:                               # EH_LABEL
# %bb.82:                               # %_ZNSolsEf.exit
.Ltmp442:                               # EH_LABEL
	movq	%rax, %r13
	.cfi_escape 0x2e, 0x00
	leaq	.L.str.4(%rip), %rsi
	movl	$6, %edx
	movq	%rax, %rdi
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
.Ltmp443:                               # EH_LABEL
# %bb.83:                               # %_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc.exit380
.Ltmp444:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movq	%r13, %rdi
	movl	%r15d, %esi
	callq	_ZNSolsEi@PLT
.Ltmp445:                               # EH_LABEL
# %bb.84:
.Ltmp446:                               # EH_LABEL
	movq	%rax, %r15
	.cfi_escape 0x2e, 0x00
	leaq	.L.str.5(%rip), %rsi
	movl	$1, %edx
	movq	%rax, %rdi
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
.Ltmp447:                               # EH_LABEL
# %bb.85:                               # %_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc.exit382
	movsd	.LCPI16_5(%rip), %xmm0          # xmm0 = [1.0E-3,0.0E+0]
	ucomisd	48(%rsp), %xmm0                 # 8-byte Folded Reload
	leaq	.L.str.6(%rip), %rax
	leaq	.L.str.7(%rip), %rsi
	cmovaq	%rax, %rsi
.Ltmp448:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movl	$4, %edx
	movq	%r15, %rdi
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
.Ltmp449:                               # EH_LABEL
# %bb.86:                               # %_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc.exit384
.Ltmp450:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	leaq	.L.str.8(%rip), %rsi
	movl	$1, %edx
	movq	%r15, %rdi
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
.Ltmp451:                               # EH_LABEL
# %bb.87:                               # %_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc.exit386
	movsd	.LCPI16_5(%rip), %xmm0          # xmm0 = [1.0E-3,0.0E+0]
	ucomisd	48(%rsp), %xmm0                 # 8-byte Folded Reload
	jbe	.LBB16_127
# %bb.88:
	movq	8(%rsp), %rdi
.Ltmp453:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movl	$2048, %edx                     # imm = 0x800
	xorl	%esi, %esi
	callq	hipMemset@PLT
.Ltmp454:                               # EH_LABEL
# %bb.89:
.Ltmp455:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movabsq	$4294967298, %rdi               # imm = 0x100000002
	movabsq	$4294967328, %rdx               # imm = 0x100000020
	movl	$1, %esi
	movl	$1, %ecx
	xorl	%r8d, %r8d
	xorl	%r9d, %r9d
	callq	__hipPushCallConfiguration@PLT
.Ltmp456:                               # EH_LABEL
# %bb.90:
	testl	%eax, %eax
	jne	.LBB16_93
# %bb.91:
	movq	88(%rsp), %rax
	movq	184(%rsp), %rcx
	movq	80(%rsp), %rdx
	movq	8(%rsp), %rsi
	movq	%rax, 168(%rsp)
	movq	%rcx, 160(%rsp)
	movq	%rdx, 152(%rsp)
	movq	%rsi, 144(%rsp)
	movl	$16, 24(%rsp)
	movl	$32, 4(%rsp)
	movl	$256, (%rsp)                    # imm = 0x100
	leaq	168(%rsp), %rax
	movq	%rax, 208(%rsp)
	leaq	160(%rsp), %rax
	movq	%rax, 216(%rsp)
	leaq	152(%rsp), %rax
	movq	%rax, 224(%rsp)
	leaq	144(%rsp), %rax
	movq	%rax, 232(%rsp)
	leaq	24(%rsp), %rax
	movq	%rax, 240(%rsp)
	leaq	4(%rsp), %rax
	movq	%rax, 248(%rsp)
	movq	%rsp, %rax
	movq	%rax, 256(%rsp)
.Ltmp457:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	leaq	128(%rsp), %rdi
	leaq	112(%rsp), %rsi
	leaq	104(%rsp), %rdx
	leaq	96(%rsp), %rcx
	callq	__hipPopCallConfiguration@PLT
.Ltmp458:                               # EH_LABEL
# %bb.92:                               # %.noexc393
	movq	128(%rsp), %rsi
	movl	136(%rsp), %edx
	movq	112(%rsp), %rcx
	movl	120(%rsp), %r8d
.Ltmp459:                               # EH_LABEL
	.cfi_escape 0x2e, 0x10
	movq	_Z24fused_m6_repacked_db_k32PK14__hip_fp8_e4m3PKhS3_Pfiii@GOTPCREL(%rip), %rdi
	leaq	208(%rsp), %r9
	pushq	96(%rsp)
	.cfi_adjust_cfa_offset 8
	pushq	112(%rsp)
	.cfi_adjust_cfa_offset 8
	callq	hipLaunchKernel@PLT
	addq	$16, %rsp
	.cfi_adjust_cfa_offset -16
.Ltmp460:                               # EH_LABEL
.LBB16_93:
.Ltmp461:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	callq	hipDeviceSynchronize@PLT
.Ltmp462:                               # EH_LABEL
# %bb.94:
	movq	8(%rsp), %rsi
.Ltmp463:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movl	$2048, %edx                     # imm = 0x800
	movq	16(%rsp), %rdi                  # 8-byte Reload
	movl	$2, %ecx
	callq	hipMemcpy@PLT
.Ltmp464:                               # EH_LABEL
# %bb.95:                               # %.preheader647.preheader
	xorps	%xmm4, %xmm4
	xorl	%r15d, %r15d
	xorl	%eax, %eax
	movq	%r14, %rdx
	movq	16(%rsp), %rsi                  # 8-byte Reload
	movsd	.LCPI16_5(%rip), %xmm2          # xmm2 = [1.0E-3,0.0E+0]
	movaps	.LCPI16_4(%rip), %xmm3          # xmm3 = [NaN,NaN,NaN,NaN]
	.p2align	4
.LBB16_96:                              # %.preheader647
                                        # =>This Inner Loop Header: Depth=1
	movaps	%xmm4, %xmm0
	movss	(%rsi,%rax,4), %xmm4            # xmm4 = mem[0],zero,zero,zero
	subss	(%rdx,%rax,4), %xmm4
	andps	%xmm3, %xmm4
	xorps	%xmm1, %xmm1
	cvtss2sd	%xmm4, %xmm1
	xorl	%ecx, %ecx
	ucomisd	%xmm2, %xmm1
	seta	%cl
	maxss	%xmm0, %xmm4
	addl	%ecx, %r15d
	incq	%rax
	cmpq	$512, %rax                      # imm = 0x200
	jne	.LBB16_96
# %bb.97:
.Ltmp465:                               # EH_LABEL
	movaps	%xmm4, 48(%rsp)                 # 16-byte Spill
	.cfi_escape 0x2e, 0x00
	movq	_ZSt4cout@GOTPCREL(%rip), %rdi
	leaq	.L.str.9(%rip), %rsi
	movl	$36, %edx
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
.Ltmp466:                               # EH_LABEL
# %bb.98:                               # %_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc.exit396
	movaps	48(%rsp), %xmm0                 # 16-byte Reload
	cvtss2sd	%xmm0, %xmm0
.Ltmp467:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movq	_ZSt4cout@GOTPCREL(%rip), %rdi
	movaps	%xmm0, 48(%rsp)                 # 16-byte Spill
	callq	_ZNSo9_M_insertIdEERSoT_@PLT
.Ltmp468:                               # EH_LABEL
# %bb.99:                               # %_ZNSolsEf.exit398
.Ltmp469:                               # EH_LABEL
	movq	%rax, %r13
	.cfi_escape 0x2e, 0x00
	leaq	.L.str.4(%rip), %rsi
	movl	$6, %edx
	movq	%rax, %rdi
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
.Ltmp470:                               # EH_LABEL
# %bb.100:                              # %_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc.exit400
.Ltmp471:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movq	%r13, %rdi
	movl	%r15d, %esi
	callq	_ZNSolsEi@PLT
.Ltmp472:                               # EH_LABEL
# %bb.101:
.Ltmp473:                               # EH_LABEL
	movq	%rax, %r15
	.cfi_escape 0x2e, 0x00
	leaq	.L.str.5(%rip), %rsi
	movl	$1, %edx
	movq	%rax, %rdi
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
.Ltmp474:                               # EH_LABEL
# %bb.102:                              # %_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc.exit402
	movsd	.LCPI16_5(%rip), %xmm0          # xmm0 = [1.0E-3,0.0E+0]
	ucomisd	48(%rsp), %xmm0                 # 16-byte Folded Reload
	leaq	.L.str.7(%rip), %rsi
	leaq	.L.str.6(%rip), %rax
	cmovaq	%rax, %rsi
.Ltmp475:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movl	$4, %edx
	movq	%r15, %rdi
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
.Ltmp476:                               # EH_LABEL
# %bb.103:                              # %_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc.exit404
.Ltmp477:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	leaq	.L.str.8(%rip), %rsi
	movl	$1, %edx
	movq	%r15, %rdi
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
.Ltmp478:                               # EH_LABEL
# %bb.104:                              # %_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc.exit406
.Ltmp479:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movl	$4096, %edi                     # imm = 0x1000
	callq	_Znwm@PLT
.Ltmp480:                               # EH_LABEL
# %bb.105:
	movq	%rax, %r15
	.cfi_escape 0x2e, 0x00
	xorl	%r13d, %r13d
	movl	$4096, %edx                     # imm = 0x1000
	movq	%rax, %rdi
	xorl	%esi, %esi
	callq	memset@PLT
	xorl	%eax, %eax
	movq	64(%rsp), %rdx                  # 8-byte Reload
	.p2align	4
.LBB16_106:                             # %.preheader.i410
                                        # =>This Inner Loop Header: Depth=1
	movq	%rax, %rcx
	shlq	$9, %rcx
	shlq	$11, %rax
	movups	(%rdx,%rax), %xmm0
	movups	%xmm0, (%r15,%rcx)
	movups	16(%rdx,%rax), %xmm0
	movups	%xmm0, 16(%r15,%rcx)
	movups	128(%rdx,%rax), %xmm0
	movups	%xmm0, 32(%r15,%rcx)
	movups	144(%rdx,%rax), %xmm0
	movups	%xmm0, 48(%r15,%rcx)
	movups	256(%rdx,%rax), %xmm0
	movups	%xmm0, 64(%r15,%rcx)
	movups	272(%rdx,%rax), %xmm0
	movups	%xmm0, 80(%r15,%rcx)
	movups	384(%rdx,%rax), %xmm0
	movups	%xmm0, 96(%r15,%rcx)
	movups	400(%rdx,%rax), %xmm0
	movups	%xmm0, 112(%r15,%rcx)
	movups	512(%rdx,%rax), %xmm0
	movups	%xmm0, 128(%r15,%rcx)
	movups	528(%rdx,%rax), %xmm0
	movups	%xmm0, 144(%r15,%rcx)
	movups	640(%rdx,%rax), %xmm0
	movups	%xmm0, 160(%r15,%rcx)
	movups	656(%rdx,%rax), %xmm0
	movups	%xmm0, 176(%r15,%rcx)
	movups	768(%rdx,%rax), %xmm0
	movups	%xmm0, 192(%r15,%rcx)
	movups	784(%rdx,%rax), %xmm0
	movups	%xmm0, 208(%r15,%rcx)
	movups	896(%rdx,%rax), %xmm0
	movups	%xmm0, 224(%r15,%rcx)
	movups	912(%rdx,%rax), %xmm0
	movups	%xmm0, 240(%r15,%rcx)
	movups	1024(%rdx,%rax), %xmm0
	movups	%xmm0, 256(%r15,%rcx)
	movups	1040(%rdx,%rax), %xmm0
	movups	%xmm0, 272(%r15,%rcx)
	movups	1152(%rdx,%rax), %xmm0
	movups	%xmm0, 288(%r15,%rcx)
	movups	1168(%rdx,%rax), %xmm0
	movups	%xmm0, 304(%r15,%rcx)
	movups	1280(%rdx,%rax), %xmm0
	movups	%xmm0, 320(%r15,%rcx)
	movups	1296(%rdx,%rax), %xmm0
	movups	%xmm0, 336(%r15,%rcx)
	movups	1408(%rdx,%rax), %xmm0
	movups	%xmm0, 352(%r15,%rcx)
	movups	1424(%rdx,%rax), %xmm0
	movups	%xmm0, 368(%r15,%rcx)
	movups	1536(%rdx,%rax), %xmm0
	movups	%xmm0, 384(%r15,%rcx)
	movups	1552(%rdx,%rax), %xmm0
	movups	%xmm0, 400(%r15,%rcx)
	movups	1664(%rdx,%rax), %xmm0
	movups	%xmm0, 416(%r15,%rcx)
	movups	1680(%rdx,%rax), %xmm0
	movups	%xmm0, 432(%r15,%rcx)
	movups	1792(%rdx,%rax), %xmm0
	movups	%xmm0, 448(%r15,%rcx)
	movups	1808(%rdx,%rax), %xmm0
	movups	%xmm0, 464(%r15,%rcx)
	movups	1920(%rdx,%rax), %xmm0
	movups	%xmm0, 480(%r15,%rcx)
	movups	1936(%rdx,%rax), %xmm0
	movups	%xmm0, 496(%r15,%rcx)
	movups	32(%rdx,%rax), %xmm0
	movups	%xmm0, 1024(%r15,%rcx)
	movups	48(%rdx,%rax), %xmm0
	movups	%xmm0, 1040(%r15,%rcx)
	movups	160(%rdx,%rax), %xmm0
	movups	%xmm0, 1056(%r15,%rcx)
	movups	176(%rdx,%rax), %xmm0
	movups	%xmm0, 1072(%r15,%rcx)
	movups	288(%rdx,%rax), %xmm0
	movups	%xmm0, 1088(%r15,%rcx)
	movups	304(%rdx,%rax), %xmm0
	movups	%xmm0, 1104(%r15,%rcx)
	movups	416(%rdx,%rax), %xmm0
	movups	%xmm0, 1120(%r15,%rcx)
	movups	432(%rdx,%rax), %xmm0
	movups	%xmm0, 1136(%r15,%rcx)
	movups	544(%rdx,%rax), %xmm0
	movups	%xmm0, 1152(%r15,%rcx)
	movups	560(%rdx,%rax), %xmm0
	movups	%xmm0, 1168(%r15,%rcx)
	movups	672(%rdx,%rax), %xmm0
	movups	%xmm0, 1184(%r15,%rcx)
	movups	688(%rdx,%rax), %xmm0
	movups	%xmm0, 1200(%r15,%rcx)
	movups	800(%rdx,%rax), %xmm0
	movups	%xmm0, 1216(%r15,%rcx)
	movups	816(%rdx,%rax), %xmm0
	movups	%xmm0, 1232(%r15,%rcx)
	movups	928(%rdx,%rax), %xmm0
	movups	%xmm0, 1248(%r15,%rcx)
	movups	944(%rdx,%rax), %xmm0
	movups	%xmm0, 1264(%r15,%rcx)
	movups	1056(%rdx,%rax), %xmm0
	movups	%xmm0, 1280(%r15,%rcx)
	movups	1072(%rdx,%rax), %xmm0
	movups	%xmm0, 1296(%r15,%rcx)
	movups	1184(%rdx,%rax), %xmm0
	movups	%xmm0, 1312(%r15,%rcx)
	movups	1200(%rdx,%rax), %xmm0
	movups	%xmm0, 1328(%r15,%rcx)
	movups	1312(%rdx,%rax), %xmm0
	movups	%xmm0, 1344(%r15,%rcx)
	movups	1328(%rdx,%rax), %xmm0
	movups	%xmm0, 1360(%r15,%rcx)
	movups	1440(%rdx,%rax), %xmm0
	movups	%xmm0, 1376(%r15,%rcx)
	movups	1456(%rdx,%rax), %xmm0
	movups	%xmm0, 1392(%r15,%rcx)
	movups	1568(%rdx,%rax), %xmm0
	movups	%xmm0, 1408(%r15,%rcx)
	movups	1584(%rdx,%rax), %xmm0
	movups	%xmm0, 1424(%r15,%rcx)
	movups	1696(%rdx,%rax), %xmm0
	movups	%xmm0, 1440(%r15,%rcx)
	movups	1712(%rdx,%rax), %xmm0
	movups	%xmm0, 1456(%r15,%rcx)
	movups	1824(%rdx,%rax), %xmm0
	movups	%xmm0, 1472(%r15,%rcx)
	movups	1840(%rdx,%rax), %xmm0
	movups	%xmm0, 1488(%r15,%rcx)
	movups	1952(%rdx,%rax), %xmm0
	movups	%xmm0, 1504(%r15,%rcx)
	movups	1968(%rdx,%rax), %xmm0
	movups	%xmm0, 1520(%r15,%rcx)
	movups	64(%rdx,%rax), %xmm0
	movups	%xmm0, 2048(%r15,%rcx)
	movups	80(%rdx,%rax), %xmm0
	movups	%xmm0, 2064(%r15,%rcx)
	movups	192(%rdx,%rax), %xmm0
	movups	%xmm0, 2080(%r15,%rcx)
	movups	208(%rdx,%rax), %xmm0
	movups	%xmm0, 2096(%r15,%rcx)
	movups	320(%rdx,%rax), %xmm0
	movups	%xmm0, 2112(%r15,%rcx)
	movups	336(%rdx,%rax), %xmm0
	movups	%xmm0, 2128(%r15,%rcx)
	movups	448(%rdx,%rax), %xmm0
	movups	%xmm0, 2144(%r15,%rcx)
	movups	464(%rdx,%rax), %xmm0
	movups	%xmm0, 2160(%r15,%rcx)
	movups	576(%rdx,%rax), %xmm0
	movups	%xmm0, 2176(%r15,%rcx)
	movups	592(%rdx,%rax), %xmm0
	movups	%xmm0, 2192(%r15,%rcx)
	movups	704(%rdx,%rax), %xmm0
	movups	%xmm0, 2208(%r15,%rcx)
	movups	720(%rdx,%rax), %xmm0
	movups	%xmm0, 2224(%r15,%rcx)
	movups	832(%rdx,%rax), %xmm0
	movups	%xmm0, 2240(%r15,%rcx)
	movups	848(%rdx,%rax), %xmm0
	movups	%xmm0, 2256(%r15,%rcx)
	movups	960(%rdx,%rax), %xmm0
	movups	%xmm0, 2272(%r15,%rcx)
	movups	976(%rdx,%rax), %xmm0
	movups	%xmm0, 2288(%r15,%rcx)
	movups	1088(%rdx,%rax), %xmm0
	movups	%xmm0, 2304(%r15,%rcx)
	movups	1104(%rdx,%rax), %xmm0
	movups	%xmm0, 2320(%r15,%rcx)
	movups	1216(%rdx,%rax), %xmm0
	movups	%xmm0, 2336(%r15,%rcx)
	movups	1232(%rdx,%rax), %xmm0
	movups	%xmm0, 2352(%r15,%rcx)
	movups	1344(%rdx,%rax), %xmm0
	movups	%xmm0, 2368(%r15,%rcx)
	movups	1360(%rdx,%rax), %xmm0
	movups	%xmm0, 2384(%r15,%rcx)
	movups	1472(%rdx,%rax), %xmm0
	movups	%xmm0, 2400(%r15,%rcx)
	movups	1488(%rdx,%rax), %xmm0
	movups	%xmm0, 2416(%r15,%rcx)
	movups	1600(%rdx,%rax), %xmm0
	movups	%xmm0, 2432(%r15,%rcx)
	movups	1616(%rdx,%rax), %xmm0
	movups	%xmm0, 2448(%r15,%rcx)
	movups	1728(%rdx,%rax), %xmm0
	movups	%xmm0, 2464(%r15,%rcx)
	movups	1744(%rdx,%rax), %xmm0
	movups	%xmm0, 2480(%r15,%rcx)
	movups	1856(%rdx,%rax), %xmm0
	movups	%xmm0, 2496(%r15,%rcx)
	movups	1872(%rdx,%rax), %xmm0
	movups	%xmm0, 2512(%r15,%rcx)
	movups	1984(%rdx,%rax), %xmm0
	movups	%xmm0, 2528(%r15,%rcx)
	movups	2000(%rdx,%rax), %xmm0
	movups	%xmm0, 2544(%r15,%rcx)
	movups	96(%rdx,%rax), %xmm0
	movups	%xmm0, 3072(%r15,%rcx)
	movups	112(%rdx,%rax), %xmm0
	movups	%xmm0, 3088(%r15,%rcx)
	movups	224(%rdx,%rax), %xmm0
	movups	%xmm0, 3104(%r15,%rcx)
	movups	240(%rdx,%rax), %xmm0
	movups	%xmm0, 3120(%r15,%rcx)
	movups	352(%rdx,%rax), %xmm0
	movups	%xmm0, 3136(%r15,%rcx)
	movups	368(%rdx,%rax), %xmm0
	movups	%xmm0, 3152(%r15,%rcx)
	movups	480(%rdx,%rax), %xmm0
	movups	%xmm0, 3168(%r15,%rcx)
	movups	496(%rdx,%rax), %xmm0
	movups	%xmm0, 3184(%r15,%rcx)
	movups	608(%rdx,%rax), %xmm0
	movups	%xmm0, 3200(%r15,%rcx)
	movups	624(%rdx,%rax), %xmm0
	movups	%xmm0, 3216(%r15,%rcx)
	movups	736(%rdx,%rax), %xmm0
	movups	%xmm0, 3232(%r15,%rcx)
	movups	752(%rdx,%rax), %xmm0
	movups	%xmm0, 3248(%r15,%rcx)
	movups	864(%rdx,%rax), %xmm0
	movups	%xmm0, 3264(%r15,%rcx)
	movups	880(%rdx,%rax), %xmm0
	movups	%xmm0, 3280(%r15,%rcx)
	movups	992(%rdx,%rax), %xmm0
	movups	%xmm0, 3296(%r15,%rcx)
	movups	1008(%rdx,%rax), %xmm0
	movups	%xmm0, 3312(%r15,%rcx)
	movups	1120(%rdx,%rax), %xmm0
	movups	%xmm0, 3328(%r15,%rcx)
	movups	1136(%rdx,%rax), %xmm0
	movups	%xmm0, 3344(%r15,%rcx)
	movups	1248(%rdx,%rax), %xmm0
	movups	%xmm0, 3360(%r15,%rcx)
	movups	1264(%rdx,%rax), %xmm0
	movups	%xmm0, 3376(%r15,%rcx)
	movups	1376(%rdx,%rax), %xmm0
	movups	%xmm0, 3392(%r15,%rcx)
	movups	1392(%rdx,%rax), %xmm0
	movups	%xmm0, 3408(%r15,%rcx)
	movups	1504(%rdx,%rax), %xmm0
	movups	%xmm0, 3424(%r15,%rcx)
	movups	1520(%rdx,%rax), %xmm0
	movups	%xmm0, 3440(%r15,%rcx)
	movups	1632(%rdx,%rax), %xmm0
	movups	%xmm0, 3456(%r15,%rcx)
	movups	1648(%rdx,%rax), %xmm0
	movups	%xmm0, 3472(%r15,%rcx)
	movups	1760(%rdx,%rax), %xmm0
	movups	%xmm0, 3488(%r15,%rcx)
	movups	1776(%rdx,%rax), %xmm0
	movups	%xmm0, 3504(%r15,%rcx)
	movups	1888(%rdx,%rax), %xmm0
	movups	%xmm0, 3520(%r15,%rcx)
	movups	1904(%rdx,%rax), %xmm0
	movups	%xmm0, 3536(%r15,%rcx)
	movups	2016(%rdx,%rax), %xmm0
	movups	%xmm0, 3552(%r15,%rcx)
	movupd	2032(%rdx,%rax), %xmm0
	movupd	%xmm0, 3568(%r15,%rcx)
	movl	$1, %eax
	testb	$1, %r13b
	movb	$1, %r13b
	je	.LBB16_106
# %bb.107:                              # %_Z17repack_k64_for_m6PKhPhiii.exit
.Ltmp482:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	leaq	24(%rsp), %rdi
	movl	$4096, %esi                     # imm = 0x1000
	callq	hipMalloc@PLT
.Ltmp483:                               # EH_LABEL
# %bb.108:                              # %_ZL9hipMallocIhE10hipError_tPPT_m.exit430
	movq	24(%rsp), %rdi
.Ltmp484:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movl	$4096, %edx                     # imm = 0x1000
	movq	%r15, %rsi
	movl	$1, %ecx
	callq	hipMemcpy@PLT
.Ltmp485:                               # EH_LABEL
# %bb.109:
	movq	8(%rsp), %rdi
.Ltmp486:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movl	$2048, %edx                     # imm = 0x800
	xorl	%esi, %esi
	callq	hipMemset@PLT
.Ltmp487:                               # EH_LABEL
# %bb.110:
.Ltmp488:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movabsq	$4294967298, %rdi               # imm = 0x100000002
	movabsq	$4294967328, %rdx               # imm = 0x100000020
	movl	$1, %esi
	movl	$1, %ecx
	xorl	%r8d, %r8d
	xorl	%r9d, %r9d
	callq	__hipPushCallConfiguration@PLT
.Ltmp489:                               # EH_LABEL
# %bb.111:
	testl	%eax, %eax
	jne	.LBB16_114
# %bb.112:
	movq	88(%rsp), %rax
	movq	24(%rsp), %rcx
	movq	80(%rsp), %rdx
	movq	8(%rsp), %rsi
	movq	%rax, 168(%rsp)
	movq	%rcx, 160(%rsp)
	movq	%rdx, 152(%rsp)
	movq	%rsi, 144(%rsp)
	movl	$16, 4(%rsp)
	movl	$32, (%rsp)
	movl	$256, 204(%rsp)                 # imm = 0x100
	leaq	168(%rsp), %rax
	movq	%rax, 208(%rsp)
	leaq	160(%rsp), %rax
	movq	%rax, 216(%rsp)
	leaq	152(%rsp), %rax
	movq	%rax, 224(%rsp)
	leaq	144(%rsp), %rax
	movq	%rax, 232(%rsp)
	leaq	4(%rsp), %rax
	movq	%rax, 240(%rsp)
	movq	%rsp, %rax
	movq	%rax, 248(%rsp)
	leaq	204(%rsp), %rax
	movq	%rax, 256(%rsp)
.Ltmp490:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	leaq	128(%rsp), %rdi
	leaq	112(%rsp), %rsi
	leaq	104(%rsp), %rdx
	leaq	96(%rsp), %rcx
	callq	__hipPopCallConfiguration@PLT
.Ltmp491:                               # EH_LABEL
# %bb.113:                              # %.noexc437
	movq	128(%rsp), %rsi
	movl	136(%rsp), %edx
	movq	112(%rsp), %rcx
	movl	120(%rsp), %r8d
.Ltmp492:                               # EH_LABEL
	.cfi_escape 0x2e, 0x10
	movq	_Z21fused_m6_repacked_k64PK14__hip_fp8_e4m3PKhS3_Pfiii@GOTPCREL(%rip), %rdi
	leaq	208(%rsp), %r9
	pushq	96(%rsp)
	.cfi_adjust_cfa_offset 8
	pushq	112(%rsp)
	.cfi_adjust_cfa_offset 8
	callq	hipLaunchKernel@PLT
	addq	$16, %rsp
	.cfi_adjust_cfa_offset -16
.Ltmp493:                               # EH_LABEL
.LBB16_114:
.Ltmp494:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	callq	hipDeviceSynchronize@PLT
.Ltmp495:                               # EH_LABEL
# %bb.115:
	movq	8(%rsp), %rsi
.Ltmp496:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movl	$2048, %edx                     # imm = 0x800
	movq	16(%rsp), %rdi                  # 8-byte Reload
	movl	$2, %ecx
	callq	hipMemcpy@PLT
.Ltmp497:                               # EH_LABEL
# %bb.116:                              # %.preheader.preheader
	xorps	%xmm4, %xmm4
	xorl	%r13d, %r13d
	xorl	%eax, %eax
	movq	%r14, %rdx
	movq	16(%rsp), %rsi                  # 8-byte Reload
	movsd	.LCPI16_5(%rip), %xmm2          # xmm2 = [1.0E-3,0.0E+0]
	movaps	.LCPI16_4(%rip), %xmm3          # xmm3 = [NaN,NaN,NaN,NaN]
	.p2align	4
.LBB16_117:                             # %.preheader
                                        # =>This Inner Loop Header: Depth=1
	movaps	%xmm4, %xmm0
	movss	(%rsi,%rax,4), %xmm4            # xmm4 = mem[0],zero,zero,zero
	subss	(%rdx,%rax,4), %xmm4
	andps	%xmm3, %xmm4
	xorps	%xmm1, %xmm1
	cvtss2sd	%xmm4, %xmm1
	xorl	%ecx, %ecx
	ucomisd	%xmm2, %xmm1
	seta	%cl
	maxss	%xmm0, %xmm4
	addl	%ecx, %r13d
	incq	%rax
	cmpq	$512, %rax                      # imm = 0x200
	jne	.LBB16_117
# %bb.118:
.Ltmp498:                               # EH_LABEL
	movaps	%xmm4, 272(%rsp)                # 16-byte Spill
	.cfi_escape 0x2e, 0x00
	movq	_ZSt4cout@GOTPCREL(%rip), %rdi
	leaq	.L.str.10(%rip), %rsi
	movl	$33, %edx
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
.Ltmp499:                               # EH_LABEL
# %bb.119:                              # %_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc.exit440
	movaps	272(%rsp), %xmm0                # 16-byte Reload
	cvtss2sd	%xmm0, %xmm0
.Ltmp500:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movq	_ZSt4cout@GOTPCREL(%rip), %rdi
	movaps	%xmm0, 272(%rsp)                # 16-byte Spill
	callq	_ZNSo9_M_insertIdEERSoT_@PLT
.Ltmp501:                               # EH_LABEL
# %bb.120:                              # %_ZNSolsEf.exit442
.Ltmp502:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	leaq	.L.str.4(%rip), %rsi
	movl	$6, %edx
	movq	%rax, %rbx
	movq	%rax, %rdi
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
.Ltmp503:                               # EH_LABEL
# %bb.121:                              # %_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc.exit444
.Ltmp504:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movq	%rbx, %rdi
	movl	%r13d, %esi
	callq	_ZNSolsEi@PLT
.Ltmp505:                               # EH_LABEL
# %bb.122:
.Ltmp506:                               # EH_LABEL
	movq	%rax, %r13
	.cfi_escape 0x2e, 0x00
	leaq	.L.str.5(%rip), %rsi
	movl	$1, %edx
	movq	%rax, %rdi
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
.Ltmp507:                               # EH_LABEL
# %bb.123:                              # %_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc.exit446
	movsd	.LCPI16_5(%rip), %xmm0          # xmm0 = [1.0E-3,0.0E+0]
	ucomisd	272(%rsp), %xmm0                # 16-byte Folded Reload
	leaq	.L.str.7(%rip), %rsi
	leaq	.L.str.6(%rip), %rax
	cmovaq	%rax, %rsi
.Ltmp508:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movl	$4, %edx
	movq	%r13, %rdi
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
.Ltmp509:                               # EH_LABEL
# %bb.124:                              # %_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc.exit448
.Ltmp510:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	leaq	.L.str.8(%rip), %rsi
	movl	$1, %edx
	movq	%r13, %rdi
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
.Ltmp511:                               # EH_LABEL
# %bb.125:                              # %_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc.exit450
	movq	24(%rsp), %rdi
.Ltmp512:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	callq	hipFree@PLT
.Ltmp513:                               # EH_LABEL
# %bb.126:                              # %_ZNSt6vectorIhSaIhEED2Ev.exit452
	movsd	.LCPI16_5(%rip), %xmm0          # xmm0 = [1.0E-3,0.0E+0]
	movapd	272(%rsp), %xmm2                # 16-byte Reload
	cmpltsd	%xmm0, %xmm2
	movapd	48(%rsp), %xmm1                 # 16-byte Reload
	cmpltsd	%xmm0, %xmm1
	andpd	%xmm2, %xmm1
	movd	%xmm1, %r13d
	.cfi_escape 0x2e, 0x00
	movl	$4096, %esi                     # imm = 0x1000
	movq	%r15, %rdi
	callq	_ZdlPvm@PLT
	jmp	.LBB16_128
.LBB16_127:
	xorl	%r13d, %r13d
.LBB16_128:
	movq	88(%rsp), %rdi
.Ltmp515:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	callq	hipFree@PLT
.Ltmp516:                               # EH_LABEL
# %bb.129:
	movq	184(%rsp), %rdi
.Ltmp517:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	callq	hipFree@PLT
.Ltmp518:                               # EH_LABEL
# %bb.130:
	movq	80(%rsp), %rdi
.Ltmp519:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	callq	hipFree@PLT
.Ltmp520:                               # EH_LABEL
# %bb.131:
	movq	8(%rsp), %rdi
.Ltmp521:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	callq	hipFree@PLT
.Ltmp522:                               # EH_LABEL
# %bb.132:                              # %_ZNSt6vectorIhSaIhEED2Ev.exit471
	.cfi_escape 0x2e, 0x00
	movl	$2048, %esi                     # imm = 0x800
	movq	16(%rsp), %rdi                  # 8-byte Reload
	callq	_ZdlPvm@PLT
	.cfi_escape 0x2e, 0x00
	movl	$4096, %esi                     # imm = 0x1000
	movq	%rbp, %rdi
	callq	_ZdlPvm@PLT
	.cfi_escape 0x2e, 0x00
	movl	$2048, %esi                     # imm = 0x800
	movq	%r14, %rdi
	callq	_ZdlPvm@PLT
	.cfi_escape 0x2e, 0x00
	movl	$32768, %esi                    # imm = 0x8000
	movq	32(%rsp), %rdi                  # 8-byte Reload
	callq	_ZdlPvm@PLT
	.cfi_escape 0x2e, 0x00
	movl	$16384, %esi                    # imm = 0x4000
	movq	40(%rsp), %rdi                  # 8-byte Reload
	callq	_ZdlPvm@PLT
	.cfi_escape 0x2e, 0x00
	movl	$8192, %esi                     # imm = 0x2000
	movq	192(%rsp), %rdi                 # 8-byte Reload
	callq	_ZdlPvm@PLT
	.cfi_escape 0x2e, 0x00
	movl	$16, %esi
	movq	288(%rsp), %rdi                 # 8-byte Reload
	callq	_ZdlPvm@PLT
	.cfi_escape 0x2e, 0x00
	movl	$16, %esi
	movq	%r12, %rdi
	callq	_ZdlPvm@PLT
	.cfi_escape 0x2e, 0x00
	movl	$4096, %esi                     # imm = 0x1000
	movq	72(%rsp), %rdi                  # 8-byte Reload
	callq	_ZdlPvm@PLT
	.cfi_escape 0x2e, 0x00
	movl	$2048, %esi                     # imm = 0x800
	movq	176(%rsp), %rdi                 # 8-byte Reload
	callq	_ZdlPvm@PLT
	.cfi_escape 0x2e, 0x00
	movl	$4096, %esi                     # imm = 0x1000
	movq	64(%rsp), %rdi                  # 8-byte Reload
	callq	_ZdlPvm@PLT
	andb	$1, %r13b
	movl	%r13d, %eax
	addq	$5320, %rsp                     # imm = 0x14C8
	.cfi_def_cfa_offset 56
	popq	%rbx
	.cfi_def_cfa_offset 48
	popq	%r12
	.cfi_def_cfa_offset 40
	popq	%r13
	.cfi_def_cfa_offset 32
	popq	%r14
	.cfi_def_cfa_offset 24
	popq	%r15
	.cfi_def_cfa_offset 16
	popq	%rbp
	.cfi_def_cfa_offset 8
	retq
.LBB16_133:
	.cfi_def_cfa_offset 5376
.Ltmp481:                               # EH_LABEL
	jmp	.LBB16_150
.LBB16_134:
.Ltmp425:                               # EH_LABEL
	jmp	.LBB16_150
.LBB16_135:
.Ltmp422:                               # EH_LABEL
	movq	%rax, %r12
	jmp	.LBB16_152
.LBB16_136:
.Ltmp393:                               # EH_LABEL
	movq	%rax, %r12
	jmp	.LBB16_153
.LBB16_137:
.Ltmp390:                               # EH_LABEL
	movq	%rax, %r12
	jmp	.LBB16_154
.LBB16_138:
.Ltmp387:                               # EH_LABEL
	movq	%rax, %r12
	jmp	.LBB16_155
.LBB16_139:
.Ltmp384:                               # EH_LABEL
	movq	%rax, %r12
	jmp	.LBB16_156
.LBB16_140:
.Ltmp381:                               # EH_LABEL
	movq	%rax, %r12
	jmp	.LBB16_157
.LBB16_141:
.Ltmp378:                               # EH_LABEL
	movq	%rax, %r12
	jmp	.LBB16_158
.LBB16_142:
.Ltmp375:                               # EH_LABEL
	jmp	.LBB16_161
.LBB16_143:
.Ltmp366:                               # EH_LABEL
	jmp	.LBB16_164
.LBB16_144:
.Ltmp360:                               # EH_LABEL
	movq	%rax, %r12
	jmp	.LBB16_166
.LBB16_145:
.Ltmp419:                               # EH_LABEL
	movq	%rax, %r12
	jmp	.LBB16_152
.LBB16_146:                             # %_ZNSt6vectorIhSaIhEED2Ev.exit
.Ltmp514:                               # EH_LABEL
	movq	%rax, %r12
	.cfi_escape 0x2e, 0x00
	movl	$4096, %esi                     # imm = 0x1000
	movq	%r15, %rdi
	callq	_ZdlPvm@PLT
	jmp	.LBB16_151
.LBB16_147:
.Ltmp410:                               # EH_LABEL
	movq	%rax, %r12
	jmp	.LBB16_152
.LBB16_148:
.Ltmp523:                               # EH_LABEL
	jmp	.LBB16_150
.LBB16_149:
.Ltmp452:                               # EH_LABEL
.LBB16_150:                             # %_ZNSt6vectorIfSaIfEED2Ev.exit473
	movq	%rax, %r12
.LBB16_151:                             # %_ZNSt6vectorIfSaIfEED2Ev.exit473
	.cfi_escape 0x2e, 0x00
	movl	$2048, %esi                     # imm = 0x800
	movq	16(%rsp), %rdi                  # 8-byte Reload
	callq	_ZdlPvm@PLT
.LBB16_152:                             # %_ZNSt6vectorIhSaIhEED2Ev.exit475
	.cfi_escape 0x2e, 0x00
	movl	$4096, %esi                     # imm = 0x1000
	movq	%rbp, %rdi
	callq	_ZdlPvm@PLT
.LBB16_153:                             # %_ZNSt6vectorIfSaIfEED2Ev.exit477
	.cfi_escape 0x2e, 0x00
	movl	$2048, %esi                     # imm = 0x800
	movq	%r14, %rdi
	callq	_ZdlPvm@PLT
.LBB16_154:                             # %_ZNSt6vectorIfSaIfEED2Ev.exit479
	.cfi_escape 0x2e, 0x00
	movl	$32768, %esi                    # imm = 0x8000
	movq	32(%rsp), %rdi                  # 8-byte Reload
	callq	_ZdlPvm@PLT
.LBB16_155:                             # %_ZNSt6vectorIfSaIfEED2Ev.exit481
	.cfi_escape 0x2e, 0x00
	movl	$16384, %esi                    # imm = 0x4000
	movq	40(%rsp), %rdi                  # 8-byte Reload
	callq	_ZdlPvm@PLT
.LBB16_156:                             # %_ZNSt6vectorIhSaIhEED2Ev.exit483
	.cfi_escape 0x2e, 0x00
	movl	$8192, %esi                     # imm = 0x2000
	movq	192(%rsp), %rdi                 # 8-byte Reload
	callq	_ZdlPvm@PLT
.LBB16_157:                             # %_ZNSt6vectorIiSaIiEED2Ev.exit485
	.cfi_escape 0x2e, 0x00
	movl	$16, %esi
	movq	288(%rsp), %rdi                 # 8-byte Reload
	callq	_ZdlPvm@PLT
.LBB16_158:                             # %_ZNSt6vectorIiSaIiEED2Ev.exit487
	.cfi_escape 0x2e, 0x00
	movl	$16, %esi
	movq	296(%rsp), %rdi                 # 8-byte Reload
	callq	_ZdlPvm@PLT
	movq	64(%rsp), %r13                  # 8-byte Reload
	movq	176(%rsp), %rbp                 # 8-byte Reload
	jmp	.LBB16_162
.LBB16_159:
.Ltmp372:                               # EH_LABEL
	jmp	.LBB16_161
.LBB16_160:
.Ltmp369:                               # EH_LABEL
.LBB16_161:                             # %_ZNSt6vectorI14__hip_fp8_e4m3SaIS0_EED2Ev.exit489
	movq	%rax, %r12
.LBB16_162:                             # %_ZNSt6vectorI14__hip_fp8_e4m3SaIS0_EED2Ev.exit489
	.cfi_escape 0x2e, 0x00
	movl	$4096, %esi                     # imm = 0x1000
	movq	72(%rsp), %rdi                  # 8-byte Reload
	callq	_ZdlPvm@PLT
	jmp	.LBB16_165
.LBB16_163:
.Ltmp363:                               # EH_LABEL
.LBB16_164:
	movq	%rax, %r12
.LBB16_165:
	.cfi_escape 0x2e, 0x00
	movl	$2048, %esi                     # imm = 0x800
	movq	%rbp, %rdi
	callq	_ZdlPvm@PLT
.LBB16_166:                             # %_ZNSt6vectorIhSaIhEED2Ev.exit493
	.cfi_escape 0x2e, 0x00
	movl	$4096, %esi                     # imm = 0x1000
	movq	%r13, %rdi
	callq	_ZdlPvm@PLT
	.cfi_escape 0x2e, 0x00
	movq	%r12, %rdi
	callq	_Unwind_Resume@PLT
.LBB16_167:
.Ltmp357:                               # EH_LABEL
	movq	%rax, %r12
	jmp	.LBB16_166
.Lfunc_end16:
	.size	_Z11validate_m6v, .Lfunc_end16-_Z11validate_m6v
	.cfi_endproc
	.section	.gcc_except_table,"a",@progbits
	.p2align	2, 0x0
GCC_except_table16:
.Lexception4:
	.byte	255                             # @LPStart Encoding = omit
	.byte	255                             # @TType Encoding = omit
	.byte	1                               # Call site Encoding = uleb128
	.uleb128 .Lcst_end4-.Lcst_begin4
.Lcst_begin4:
	.uleb128 .Lfunc_begin4-.Lfunc_begin4    # >> Call Site 1 <<
	.uleb128 .Ltmp355-.Lfunc_begin4         #   Call between .Lfunc_begin4 and .Ltmp355
	.byte	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp355-.Lfunc_begin4         # >> Call Site 2 <<
	.uleb128 .Ltmp356-.Ltmp355              #   Call between .Ltmp355 and .Ltmp356
	.uleb128 .Ltmp357-.Lfunc_begin4         #     jumps to .Ltmp357
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp358-.Lfunc_begin4         # >> Call Site 3 <<
	.uleb128 .Ltmp359-.Ltmp358              #   Call between .Ltmp358 and .Ltmp359
	.uleb128 .Ltmp360-.Lfunc_begin4         #     jumps to .Ltmp360
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp359-.Lfunc_begin4         # >> Call Site 4 <<
	.uleb128 .Ltmp361-.Ltmp359              #   Call between .Ltmp359 and .Ltmp361
	.byte	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp361-.Lfunc_begin4         # >> Call Site 5 <<
	.uleb128 .Ltmp362-.Ltmp361              #   Call between .Ltmp361 and .Ltmp362
	.uleb128 .Ltmp363-.Lfunc_begin4         #     jumps to .Ltmp363
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp364-.Lfunc_begin4         # >> Call Site 6 <<
	.uleb128 .Ltmp365-.Ltmp364              #   Call between .Ltmp364 and .Ltmp365
	.uleb128 .Ltmp366-.Lfunc_begin4         #     jumps to .Ltmp366
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp365-.Lfunc_begin4         # >> Call Site 7 <<
	.uleb128 .Ltmp367-.Ltmp365              #   Call between .Ltmp365 and .Ltmp367
	.byte	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp367-.Lfunc_begin4         # >> Call Site 8 <<
	.uleb128 .Ltmp368-.Ltmp367              #   Call between .Ltmp367 and .Ltmp368
	.uleb128 .Ltmp369-.Lfunc_begin4         #     jumps to .Ltmp369
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp370-.Lfunc_begin4         # >> Call Site 9 <<
	.uleb128 .Ltmp371-.Ltmp370              #   Call between .Ltmp370 and .Ltmp371
	.uleb128 .Ltmp372-.Lfunc_begin4         #     jumps to .Ltmp372
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp373-.Lfunc_begin4         # >> Call Site 10 <<
	.uleb128 .Ltmp374-.Ltmp373              #   Call between .Ltmp373 and .Ltmp374
	.uleb128 .Ltmp375-.Lfunc_begin4         #     jumps to .Ltmp375
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp376-.Lfunc_begin4         # >> Call Site 11 <<
	.uleb128 .Ltmp377-.Ltmp376              #   Call between .Ltmp376 and .Ltmp377
	.uleb128 .Ltmp378-.Lfunc_begin4         #     jumps to .Ltmp378
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp379-.Lfunc_begin4         # >> Call Site 12 <<
	.uleb128 .Ltmp380-.Ltmp379              #   Call between .Ltmp379 and .Ltmp380
	.uleb128 .Ltmp381-.Lfunc_begin4         #     jumps to .Ltmp381
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp380-.Lfunc_begin4         # >> Call Site 13 <<
	.uleb128 .Ltmp382-.Ltmp380              #   Call between .Ltmp380 and .Ltmp382
	.byte	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp382-.Lfunc_begin4         # >> Call Site 14 <<
	.uleb128 .Ltmp383-.Ltmp382              #   Call between .Ltmp382 and .Ltmp383
	.uleb128 .Ltmp384-.Lfunc_begin4         #     jumps to .Ltmp384
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp383-.Lfunc_begin4         # >> Call Site 15 <<
	.uleb128 .Ltmp385-.Ltmp383              #   Call between .Ltmp383 and .Ltmp385
	.byte	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp385-.Lfunc_begin4         # >> Call Site 16 <<
	.uleb128 .Ltmp386-.Ltmp385              #   Call between .Ltmp385 and .Ltmp386
	.uleb128 .Ltmp387-.Lfunc_begin4         #     jumps to .Ltmp387
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp386-.Lfunc_begin4         # >> Call Site 17 <<
	.uleb128 .Ltmp388-.Ltmp386              #   Call between .Ltmp386 and .Ltmp388
	.byte	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp388-.Lfunc_begin4         # >> Call Site 18 <<
	.uleb128 .Ltmp389-.Ltmp388              #   Call between .Ltmp388 and .Ltmp389
	.uleb128 .Ltmp390-.Lfunc_begin4         #     jumps to .Ltmp390
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp389-.Lfunc_begin4         # >> Call Site 19 <<
	.uleb128 .Ltmp391-.Ltmp389              #   Call between .Ltmp389 and .Ltmp391
	.byte	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp391-.Lfunc_begin4         # >> Call Site 20 <<
	.uleb128 .Ltmp392-.Ltmp391              #   Call between .Ltmp391 and .Ltmp392
	.uleb128 .Ltmp393-.Lfunc_begin4         #     jumps to .Ltmp393
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp392-.Lfunc_begin4         # >> Call Site 21 <<
	.uleb128 .Ltmp394-.Ltmp392              #   Call between .Ltmp392 and .Ltmp394
	.byte	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp394-.Lfunc_begin4         # >> Call Site 22 <<
	.uleb128 .Ltmp409-.Ltmp394              #   Call between .Ltmp394 and .Ltmp409
	.uleb128 .Ltmp410-.Lfunc_begin4         #     jumps to .Ltmp410
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp411-.Lfunc_begin4         # >> Call Site 23 <<
	.uleb128 .Ltmp418-.Ltmp411              #   Call between .Ltmp411 and .Ltmp418
	.uleb128 .Ltmp419-.Lfunc_begin4         #     jumps to .Ltmp419
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp420-.Lfunc_begin4         # >> Call Site 24 <<
	.uleb128 .Ltmp421-.Ltmp420              #   Call between .Ltmp420 and .Ltmp421
	.uleb128 .Ltmp422-.Lfunc_begin4         #     jumps to .Ltmp422
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp421-.Lfunc_begin4         # >> Call Site 25 <<
	.uleb128 .Ltmp423-.Ltmp421              #   Call between .Ltmp421 and .Ltmp423
	.byte	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp423-.Lfunc_begin4         # >> Call Site 26 <<
	.uleb128 .Ltmp424-.Ltmp423              #   Call between .Ltmp423 and .Ltmp424
	.uleb128 .Ltmp425-.Lfunc_begin4         #     jumps to .Ltmp425
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp426-.Lfunc_begin4         # >> Call Site 27 <<
	.uleb128 .Ltmp451-.Ltmp426              #   Call between .Ltmp426 and .Ltmp451
	.uleb128 .Ltmp452-.Lfunc_begin4         #     jumps to .Ltmp452
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp453-.Lfunc_begin4         # >> Call Site 28 <<
	.uleb128 .Ltmp478-.Ltmp453              #   Call between .Ltmp453 and .Ltmp478
	.uleb128 .Ltmp523-.Lfunc_begin4         #     jumps to .Ltmp523
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp479-.Lfunc_begin4         # >> Call Site 29 <<
	.uleb128 .Ltmp480-.Ltmp479              #   Call between .Ltmp479 and .Ltmp480
	.uleb128 .Ltmp481-.Lfunc_begin4         #     jumps to .Ltmp481
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp480-.Lfunc_begin4         # >> Call Site 30 <<
	.uleb128 .Ltmp482-.Ltmp480              #   Call between .Ltmp480 and .Ltmp482
	.byte	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp482-.Lfunc_begin4         # >> Call Site 31 <<
	.uleb128 .Ltmp513-.Ltmp482              #   Call between .Ltmp482 and .Ltmp513
	.uleb128 .Ltmp514-.Lfunc_begin4         #     jumps to .Ltmp514
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp515-.Lfunc_begin4         # >> Call Site 32 <<
	.uleb128 .Ltmp522-.Ltmp515              #   Call between .Ltmp515 and .Ltmp522
	.uleb128 .Ltmp523-.Lfunc_begin4         #     jumps to .Ltmp523
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp522-.Lfunc_begin4         # >> Call Site 33 <<
	.uleb128 .Lfunc_end16-.Ltmp522          #   Call between .Ltmp522 and .Lfunc_end16
	.byte	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
.Lcst_end4:
	.p2align	2, 0x0
                                        # -- End function
	.section	.rodata.cst8,"aM",@progbits,8
	.p2align	3, 0x0                          # -- Begin function main
.LCPI17_0:
	.quad	0x3fef0a3d70a3d70a              # double 0.96999999999999997
.LCPI17_1:
	.quad	0x408f400000000000              # double 1000
.LCPI17_2:
	.quad	0x426d1a94a2000000              # double 1.0E+12
.LCPI17_3:
	.quad	0x41cdcd6500000000              # double 1.0E+9
.LCPI17_4:
	.quad	0x4170000000000000              # double 16777216
.LCPI17_5:
	.quad	0x4160000000000000              # double 8388608
.LCPI17_6:
	.quad	0x4080280000000000              # double 517
.LCPI17_7:
	.quad	0x3f909d6e0d93fb98              # double 0.016225547388781431
.LCPI17_8:
	.quad	0x412e848000000000              # double 1.0E+6
.LCPI17_9:
	.quad	0x4034000000000000              # double 20
	.text
	.globl	main
	.prefalign	4, .Lfunc_end17, nop
	.type	main,@function
main:                                   # @main
.Lfunc_begin5:
	.cfi_startproc
	.cfi_personality 155, DW.ref.__gxx_personality_v0
	.cfi_lsda 27, .Lexception5
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	pushq	%r15
	.cfi_def_cfa_offset 24
	pushq	%r14
	.cfi_def_cfa_offset 32
	pushq	%r13
	.cfi_def_cfa_offset 40
	pushq	%r12
	.cfi_def_cfa_offset 48
	pushq	%rbx
	.cfi_def_cfa_offset 56
	subq	$1592, %rsp                     # imm = 0x638
	.cfi_def_cfa_offset 1648
	.cfi_offset %rbx, -56
	.cfi_offset %r12, -48
	.cfi_offset %r13, -40
	.cfi_offset %r14, -32
	.cfi_offset %r15, -24
	.cfi_offset %rbp, -16
	leaq	120(%rsp), %r14
	movq	%r14, %rdi
	xorl	%esi, %esi
	callq	hipGetDevicePropertiesR0600@PLT
	movq	_ZSt4cout@GOTPCREL(%rip), %rbx
	leaq	.L.str.11(%rip), %rsi
	movl	$7, %edx
	movq	%rbx, %rdi
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
	movq	%r14, %rdi
	callq	strlen@PLT
	movq	%rbx, %rdi
	movq	%r14, %rsi
	movq	%rax, %rdx
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
	leaq	.L.str.5(%rip), %r14
	movl	$1, %edx
	movq	%rbx, %rdi
	movq	%r14, %rsi
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
	leaq	1280(%rsp), %r15
	movq	%r15, %rdi
	callq	strlen@PLT
	movq	%rbx, %rdi
	movq	%r15, %rsi
	movq	%rax, %rdx
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
	movl	$1, %edx
	movq	%rbx, %rdi
	movq	%r14, %rsi
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
	movl	508(%rsp), %esi
	movq	%rbx, %rdi
	callq	_ZNSolsEi@PLT
	leaq	.L.str.12(%rip), %rsi
	movl	$5, %edx
	movq	%rax, %rdi
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
	callq	_Z11validate_m6v
	testb	%al, %al
	jne	.LBB17_2
# %bb.1:
	movq	_ZSt4cout@GOTPCREL(%rip), %rdi
	leaq	.L.str.13(%rip), %rsi
	movl	$51, %edx
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
.LBB17_2:
	movl	$72, %edi
	callq	_Znwm@PLT
	movabsq	$4398046513152, %rcx            # imm = 0x40000000800
	movq	%rcx, 64(%rax)
	movups	.Lconstinit(%rip), %xmm0
	movups	%xmm0, (%rax)
	movups	.Lconstinit+16(%rip), %xmm0
	movups	%xmm0, 16(%rax)
	movups	.Lconstinit+32(%rip), %xmm0
	movups	%xmm0, 32(%rax)
	movupd	.Lconstinit+48(%rip), %xmm0
	movq	%rax, 104(%rsp)                 # 8-byte Spill
	movupd	%xmm0, 48(%rax)
.Ltmp524:                               # EH_LABEL
	movq	_ZSt4cout@GOTPCREL(%rip), %rdi
	leaq	.L.str.14(%rip), %rsi
	movl	$82, %edx
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
.Ltmp525:                               # EH_LABEL
# %bb.3:                                # %_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc.exit.preheader.preheader
	xorl	%ecx, %ecx
	.p2align	4
.LBB17_4:                               # %_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc.exit.preheader
                                        # =>This Loop Header: Depth=1
                                        #     Child Loop BB17_7 Depth 2
                                        #     Child Loop BB17_45 Depth 2
                                        #     Child Loop BB17_50 Depth 2
	movq	104(%rsp), %rax                 # 8-byte Reload
	movl	(%rax,%rcx), %edx
	movl	%edx, (%rsp)                    # 4-byte Spill
	movl	4(%rax,%rcx), %edx
	movl	%edx, 40(%rsp)                  # 4-byte Spill
	movq	%rcx, 72(%rsp)                  # 8-byte Spill
	movl	8(%rax,%rcx), %eax
	movl	%eax, 32(%rsp)                  # 4-byte Spill
	movl	$20, %r14d
	xorl	%r15d, %r15d
	xorl	%r12d, %r12d
	movq	$0, 24(%rsp)                    # 8-byte Folded Spill
	xorl	%r13d, %r13d
	xorl	%ebx, %ebx
	movq	$0, 16(%rsp)                    # 8-byte Folded Spill
	jmp	.LBB17_7
	.p2align	4
.LBB17_5:                               #   in Loop: Header=BB17_7 Depth=2
	movsd	%xmm0, (%rbx)
	movq	8(%rsp), %r13                   # 8-byte Reload
	movq	48(%rsp), %rax                  # 8-byte Reload
.LBB17_6:                               # %_ZNSt6vectorIdSaIdEE9push_backEOd.exit251
                                        #   in Loop: Header=BB17_7 Depth=2
	leaq	8(%rax), %r12
	addq	$8, %rbx
	decl	%r14d
	movq	80(%rsp), %r15                  # 8-byte Reload
	je	.LBB17_26
.LBB17_7:                               #   Parent Loop BB17_4 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
.Ltmp526:                               # EH_LABEL
	movq	%r13, 8(%rsp)                   # 8-byte Spill
	movl	(%rsp), %edi                    # 4-byte Reload
	movl	40(%rsp), %esi                  # 4-byte Reload
	movl	32(%rsp), %edx                  # 4-byte Reload
	xorl	%ecx, %ecx
	movl	$300, %r8d                      # imm = 0x12C
	callq	_Z11bench_plainiiiii
.Ltmp527:                               # EH_LABEL
# %bb.8:                                #   in Loop: Header=BB17_7 Depth=2
	cmpq	24(%rsp), %r12                  # 8-byte Folded Reload
	je	.LBB17_10
# %bb.9:                                #   in Loop: Header=BB17_7 Depth=2
	movq	%r15, 80(%rsp)                  # 8-byte Spill
	movsd	%xmm0, (%r12)
	movq	%r12, 48(%rsp)                  # 8-byte Spill
	jmp	.LBB17_17
	.p2align	4
.LBB17_10:                              #   in Loop: Header=BB17_7 Depth=2
	movq	%r12, %rbp
	subq	%r15, %rbp
	movabsq	$9223372036854775800, %rax      # imm = 0x7FFFFFFFFFFFFFF8
	cmpq	%rax, %rbp
	je	.LBB17_373
# %bb.11:                               # %_ZNKSt6vectorIdSaIdEE12_M_check_lenEmPKc.exit.i.i.i
                                        #   in Loop: Header=BB17_7 Depth=2
	movq	%rbp, %r13
	sarq	$3, %r13
	cmpq	$1, %r13
	adcq	%r13, %r13
	movabsq	$1152921504606846975, %rax      # imm = 0xFFFFFFFFFFFFFFF
	cmpq	%rax, %r13
	cmovaeq	%rax, %r13
	leaq	(,%r13,8), %rdi
.Ltmp528:                               # EH_LABEL
	movq	%r12, 24(%rsp)                  # 8-byte Spill
	movsd	%xmm0, 48(%rsp)                 # 8-byte Spill
	callq	_Znwm@PLT
	movsd	48(%rsp), %xmm0                 # 8-byte Reload
                                        # xmm0 = mem[0],zero
.Ltmp529:                               # EH_LABEL
# %bb.12:                               # %.noexc241
                                        #   in Loop: Header=BB17_7 Depth=2
	movq	%rax, %r12
	movsd	%xmm0, (%rax,%rbp)
	testq	%rbp, %rbp
	jle	.LBB17_14
# %bb.13:                               #   in Loop: Header=BB17_7 Depth=2
	movq	%r12, %rdi
	movq	%r15, %rsi
	movq	%rbp, %rdx
	callq	memcpy@PLT
.LBB17_14:                              # %_ZNSt6vectorIdSaIdEE11_S_relocateEPdS2_S2_RS0_.exit.i.i.i
                                        #   in Loop: Header=BB17_7 Depth=2
	testq	%r15, %r15
	je	.LBB17_16
# %bb.15:                               # %_ZNSt12_Vector_baseIdSaIdEE13_M_deallocateEPdm.exit.i.i.i.i
                                        #   in Loop: Header=BB17_7 Depth=2
	movq	%r15, %rdi
	movq	%rbp, %rsi
	callq	_ZdlPvm@PLT
.LBB17_16:                              # %_ZNSt6vectorIdSaIdEE17_M_realloc_appendIJdEEEvDpOT_.exit.i.i
                                        #   in Loop: Header=BB17_7 Depth=2
	addq	%r12, %rbp
	movq	%rbp, 48(%rsp)                  # 8-byte Spill
	leaq	(%r12,%r13,8), %rax
	movq	%rax, 24(%rsp)                  # 8-byte Spill
	movq	%r12, 80(%rsp)                  # 8-byte Spill
.LBB17_17:                              # %_ZNSt6vectorIdSaIdEE9push_backEOd.exit
                                        #   in Loop: Header=BB17_7 Depth=2
.Ltmp531:                               # EH_LABEL
	movl	(%rsp), %edi                    # 4-byte Reload
	movl	40(%rsp), %esi                  # 4-byte Reload
	movl	32(%rsp), %edx                  # 4-byte Reload
	movl	$1, %ecx
	movl	$300, %r8d                      # imm = 0x12C
	callq	_Z11bench_plainiiiii
.Ltmp532:                               # EH_LABEL
# %bb.18:                               #   in Loop: Header=BB17_7 Depth=2
	cmpq	16(%rsp), %rbx                  # 8-byte Folded Reload
	jne	.LBB17_5
# %bb.19:                               #   in Loop: Header=BB17_7 Depth=2
	movq	%rbx, %r12
	movq	8(%rsp), %r13                   # 8-byte Reload
	subq	%r13, %r12
	movabsq	$9223372036854775800, %rax      # imm = 0x7FFFFFFFFFFFFFF8
	cmpq	%rax, %r12
	je	.LBB17_375
# %bb.20:                               # %_ZNKSt6vectorIdSaIdEE12_M_check_lenEmPKc.exit.i.i.i243
                                        #   in Loop: Header=BB17_7 Depth=2
	movq	%r12, %rbp
	sarq	$3, %rbp
	cmpq	$1, %rbp
	adcq	%rbp, %rbp
	movabsq	$1152921504606846975, %rax      # imm = 0xFFFFFFFFFFFFFFF
	cmpq	%rax, %rbp
	cmovaeq	%rax, %rbp
	leaq	(,%rbp,8), %rdi
.Ltmp533:                               # EH_LABEL
	movq	%rbx, 16(%rsp)                  # 8-byte Spill
	movsd	%xmm0, 56(%rsp)                 # 8-byte Spill
	callq	_Znwm@PLT
	movsd	56(%rsp), %xmm0                 # 8-byte Reload
                                        # xmm0 = mem[0],zero
.Ltmp534:                               # EH_LABEL
# %bb.21:                               # %.noexc250
                                        #   in Loop: Header=BB17_7 Depth=2
	movq	%rax, %r13
	movsd	%xmm0, (%rax,%r12)
	testq	%r12, %r12
	movq	8(%rsp), %rbx                   # 8-byte Reload
	jle	.LBB17_23
# %bb.22:                               #   in Loop: Header=BB17_7 Depth=2
	movq	%r13, %rdi
	movq	%rbx, %rsi
	movq	%r12, %rdx
	callq	memcpy@PLT
.LBB17_23:                              # %_ZNSt6vectorIdSaIdEE11_S_relocateEPdS2_S2_RS0_.exit.i.i.i245
                                        #   in Loop: Header=BB17_7 Depth=2
	testq	%rbx, %rbx
	movq	48(%rsp), %r15                  # 8-byte Reload
	je	.LBB17_25
# %bb.24:                               # %_ZNSt12_Vector_baseIdSaIdEE13_M_deallocateEPdm.exit.i.i.i.i247
                                        #   in Loop: Header=BB17_7 Depth=2
	movq	%rbx, %rdi
	movq	%r12, %rsi
	callq	_ZdlPvm@PLT
.LBB17_25:                              # %_ZNSt6vectorIdSaIdEE17_M_realloc_appendIJdEEEvDpOT_.exit.i.i248
                                        #   in Loop: Header=BB17_7 Depth=2
	addq	%r13, %r12
	leaq	(,%rbp,8), %rax
	addq	%r13, %rax
	movq	%rax, 16(%rsp)                  # 8-byte Spill
	movq	%r12, %rbx
	movq	%r15, %rax
	jmp	.LBB17_6
	.p2align	4
.LBB17_26:                              #   in Loop: Header=BB17_4 Depth=1
	movq	%rax, 48(%rsp)                  # 8-byte Spill
	movq	%r15, %rbp
	cmpq	%r12, %r15
	je	.LBB17_29
# %bb.27:                               #   in Loop: Header=BB17_4 Depth=1
	movq	%r12, %r14
	subq	%rbp, %r14
	sarq	$3, %r14
	bsrq	%r14, %rdx
	xorl	$63, %edx
	addl	%edx, %edx
	xorq	$126, %rdx
.Ltmp536:                               # EH_LABEL
	movq	%rbp, %rdi
	movq	%r12, %rsi
	callq	_ZSt16__introsort_loopIN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEElNS0_5__ops15_Iter_less_iterEEvT_S9_T0_T1_
.Ltmp537:                               # EH_LABEL
# %bb.28:                               # %.noexc
                                        #   in Loop: Header=BB17_4 Depth=1
.Ltmp538:                               # EH_LABEL
	movq	%rbp, %rdi
	movq	%r12, %rsi
	callq	_ZSt22__final_insertion_sortIN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEENS0_5__ops15_Iter_less_iterEEvT_S9_T0_
.Ltmp539:                               # EH_LABEL
	jmp	.LBB17_30
	.p2align	4
.LBB17_29:                              # %._crit_edge1918
                                        #   in Loop: Header=BB17_4 Depth=1
	xorl	%r14d, %r14d
.LBB17_30:                              #   in Loop: Header=BB17_4 Depth=1
	andq	$-2, %r14
	movq	%rbp, %r15
	movsd	(%rbp,%r14,4), %xmm0            # xmm0 = mem[0],zero
	movsd	%xmm0, 8(%rsp)                  # 8-byte Spill
	cmpq	%rbx, %r13
	je	.LBB17_33
# %bb.31:                               #   in Loop: Header=BB17_4 Depth=1
	movq	%rbx, %r14
	subq	%r13, %r14
	sarq	$3, %r14
	bsrq	%r14, %rdx
	xorl	$63, %edx
	addl	%edx, %edx
	xorq	$126, %rdx
.Ltmp541:                               # EH_LABEL
	movq	%r13, %rdi
	movq	%rbx, %rsi
	callq	_ZSt16__introsort_loopIN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEElNS0_5__ops15_Iter_less_iterEEvT_S9_T0_T1_
.Ltmp542:                               # EH_LABEL
# %bb.32:                               # %.noexc255
                                        #   in Loop: Header=BB17_4 Depth=1
.Ltmp543:                               # EH_LABEL
	movq	%r13, %rdi
	movq	%rbx, %rsi
	callq	_ZSt22__final_insertion_sortIN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEENS0_5__ops15_Iter_less_iterEEvT_S9_T0_
.Ltmp544:                               # EH_LABEL
	jmp	.LBB17_34
	.p2align	4
.LBB17_33:                              # %._crit_edge1917
                                        #   in Loop: Header=BB17_4 Depth=1
	xorl	%r14d, %r14d
.LBB17_34:                              #   in Loop: Header=BB17_4 Depth=1
	andq	$-2, %r14
	movsd	(%r13,%r14,4), %xmm0            # xmm0 = mem[0],zero
	movsd	%xmm0, 80(%rsp)                 # 8-byte Spill
.Ltmp545:                               # EH_LABEL
	movl	$1, %edx
	movq	_ZSt4cout@GOTPCREL(%rip), %rdi
	leaq	.L.str.15(%rip), %rsi
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
.Ltmp546:                               # EH_LABEL
# %bb.35:                               # %_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc.exit259
                                        #   in Loop: Header=BB17_4 Depth=1
.Ltmp547:                               # EH_LABEL
	movq	_ZSt4cout@GOTPCREL(%rip), %rdi
	movl	(%rsp), %esi                    # 4-byte Reload
	callq	_ZNSolsEi@PLT
.Ltmp548:                               # EH_LABEL
# %bb.36:                               #   in Loop: Header=BB17_4 Depth=1
.Ltmp549:                               # EH_LABEL
	movq	%rax, %rbx
	movl	$2, %edx
	movq	%rax, %rdi
	leaq	.L.str.1(%rip), %rsi
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
.Ltmp550:                               # EH_LABEL
# %bb.37:                               # %_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc.exit261
                                        #   in Loop: Header=BB17_4 Depth=1
.Ltmp551:                               # EH_LABEL
	movq	%rbx, %rdi
	movl	40(%rsp), %esi                  # 4-byte Reload
	callq	_ZNSolsEi@PLT
.Ltmp552:                               # EH_LABEL
# %bb.38:                               #   in Loop: Header=BB17_4 Depth=1
.Ltmp553:                               # EH_LABEL
	movq	%rax, %rbx
	movl	$2, %edx
	movq	%rax, %rdi
	leaq	.L.str.2(%rip), %rsi
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
.Ltmp554:                               # EH_LABEL
# %bb.39:                               # %_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc.exit263
                                        #   in Loop: Header=BB17_4 Depth=1
.Ltmp555:                               # EH_LABEL
	movq	%rbx, %rdi
	movl	32(%rsp), %esi                  # 4-byte Reload
	callq	_ZNSolsEi@PLT
.Ltmp556:                               # EH_LABEL
# %bb.40:                               #   in Loop: Header=BB17_4 Depth=1
.Ltmp557:                               # EH_LABEL
	movq	%rax, %rbx
	movl	$16, %edx
	movq	%rax, %rdi
	leaq	.L.str.16(%rip), %rsi
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
.Ltmp558:                               # EH_LABEL
# %bb.41:                               # %_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc.exit265
                                        #   in Loop: Header=BB17_4 Depth=1
.Ltmp559:                               # EH_LABEL
	movq	%rbx, %rdi
	movsd	8(%rsp), %xmm0                  # 8-byte Reload
                                        # xmm0 = mem[0],zero
	callq	_ZNSo9_M_insertIdEERSoT_@PLT
.Ltmp560:                               # EH_LABEL
# %bb.42:                               # %_ZNSolsEd.exit
                                        #   in Loop: Header=BB17_4 Depth=1
.Ltmp561:                               # EH_LABEL
	movq	%rax, %rbx
	movl	$8, %edx
	movq	%rax, %rdi
	leaq	.L.str.17(%rip), %rsi
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
.Ltmp562:                               # EH_LABEL
# %bb.43:                               # %_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc.exit268
                                        #   in Loop: Header=BB17_4 Depth=1
	cmpq	%r12, %r15
	sete	%al
	leaq	8(%r15), %r14
	movq	48(%rsp), %r12                  # 8-byte Reload
	cmpq	%r12, %r15
	sete	%bpl
	orb	%al, %bpl
	movq	%r15, %rax
	jne	.LBB17_46
# %bb.44:                               # %.lr.ph.preheader.i.i
                                        #   in Loop: Header=BB17_4 Depth=1
	movq	%r15, %rax
	movsd	(%r15), %xmm0                   # xmm0 = mem[0],zero
	movq	%r14, %rcx
	.p2align	4
.LBB17_45:                              # %.lr.ph.i.i
                                        #   Parent Loop BB17_4 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	movapd	%xmm0, %xmm1
	movsd	(%rcx), %xmm0                   # xmm0 = mem[0],zero
	ucomisd	%xmm0, %xmm1
	cmovaq	%rcx, %rax
	minsd	%xmm1, %xmm0
	cmpq	%r12, %rcx
	leaq	8(%rcx), %rcx
	jne	.LBB17_45
.LBB17_46:                              # %_ZSt11min_elementIN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEEET_S7_S7_.exit
                                        #   in Loop: Header=BB17_4 Depth=1
	movsd	(%rax), %xmm0                   # xmm0 = mem[0],zero
.Ltmp564:                               # EH_LABEL
	movq	%rbx, %rdi
	callq	_ZNSo9_M_insertIdEERSoT_@PLT
.Ltmp565:                               # EH_LABEL
# %bb.47:                               # %_ZNSolsEd.exit271
                                        #   in Loop: Header=BB17_4 Depth=1
.Ltmp566:                               # EH_LABEL
	movq	%rax, %rbx
	movl	$5, %edx
	movq	%rax, %rdi
	leaq	.L.str.18(%rip), %rsi
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
.Ltmp567:                               # EH_LABEL
# %bb.48:                               # %_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc.exit273
                                        #   in Loop: Header=BB17_4 Depth=1
	movq	%r15, %rax
	testb	%bpl, %bpl
	jne	.LBB17_51
# %bb.49:                               # %.lr.ph.preheader.i.i276
                                        #   in Loop: Header=BB17_4 Depth=1
	movq	%r15, %rax
	movsd	(%r15), %xmm0                   # xmm0 = mem[0],zero
	.p2align	4
.LBB17_50:                              # %.lr.ph.i.i278
                                        #   Parent Loop BB17_4 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	movapd	%xmm0, %xmm1
	movsd	(%r14), %xmm0                   # xmm0 = mem[0],zero
	ucomisd	%xmm1, %xmm0
	cmovaq	%r14, %rax
	maxsd	%xmm1, %xmm0
	cmpq	%r12, %r14
	leaq	8(%r14), %r14
	jne	.LBB17_50
.LBB17_51:                              # %_ZSt11max_elementIN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEEET_S7_S7_.exit
                                        #   in Loop: Header=BB17_4 Depth=1
	movsd	(%rax), %xmm0                   # xmm0 = mem[0],zero
.Ltmp569:                               # EH_LABEL
	movq	%rbx, %rdi
	callq	_ZNSo9_M_insertIdEERSoT_@PLT
.Ltmp570:                               # EH_LABEL
# %bb.52:                               # %_ZNSolsEd.exit284
                                        #   in Loop: Header=BB17_4 Depth=1
.Ltmp571:                               # EH_LABEL
	movq	%rax, %rbx
	movl	$19, %edx
	movq	%rax, %rdi
	leaq	.L.str.19(%rip), %rsi
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
.Ltmp572:                               # EH_LABEL
# %bb.53:                               # %_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc.exit286
                                        #   in Loop: Header=BB17_4 Depth=1
.Ltmp573:                               # EH_LABEL
	movq	%rbx, %rdi
	movsd	80(%rsp), %xmm0                 # 8-byte Reload
                                        # xmm0 = mem[0],zero
	callq	_ZNSo9_M_insertIdEERSoT_@PLT
.Ltmp574:                               # EH_LABEL
# %bb.54:                               # %_ZNSolsEd.exit288
                                        #   in Loop: Header=BB17_4 Depth=1
.Ltmp575:                               # EH_LABEL
	movl	$3, %edx
	movq	%rax, %rdi
	leaq	.L.str.20(%rip), %rsi
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
.Ltmp576:                               # EH_LABEL
# %bb.55:                               # %_ZNSt6vectorIdSaIdEED2Ev.exit293
                                        #   in Loop: Header=BB17_4 Depth=1
	movq	16(%rsp), %rsi                  # 8-byte Reload
	subq	%r13, %rsi
	movq	%r13, %rdi
	callq	_ZdlPvm@PLT
	movq	24(%rsp), %rsi                  # 8-byte Reload
	subq	%r15, %rsi
	movq	%r15, %rdi
	callq	_ZdlPvm@PLT
	movq	72(%rsp), %rcx                  # 8-byte Reload
	addq	$12, %rcx
	cmpq	$72, %rcx
	jne	.LBB17_4
# %bb.56:
.Ltmp578:                               # EH_LABEL
	movq	_ZSt4cout@GOTPCREL(%rip), %rdi
	leaq	.L.str.21(%rip), %rsi
	movl	$75, %edx
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
.Ltmp579:                               # EH_LABEL
# %bb.57:                               # %_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc.exit238.preheader.preheader
	xorl	%ecx, %ecx
	.p2align	4
.LBB17_58:                              # %_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc.exit238.preheader
                                        # =>This Loop Header: Depth=1
                                        #     Child Loop BB17_61 Depth 2
	movq	104(%rsp), %rax                 # 8-byte Reload
	movl	(%rax,%rcx), %edx
	movl	%edx, 96(%rsp)                  # 4-byte Spill
	movl	4(%rax,%rcx), %r12d
	movq	%rcx, 112(%rsp)                 # 8-byte Spill
	movl	8(%rax,%rcx), %ebx
	movl	$20, %eax
	xorl	%r14d, %r14d
	movq	$0, 80(%rsp)                    # 8-byte Folded Spill
	movq	$0, 48(%rsp)                    # 8-byte Folded Spill
	xorl	%r13d, %r13d
	xorl	%ebp, %ebp
	movq	$0, 16(%rsp)                    # 8-byte Folded Spill
	movq	$0, 88(%rsp)                    # 8-byte Folded Spill
	movq	$0, (%rsp)                      # 8-byte Folded Spill
	movq	$0, 56(%rsp)                    # 8-byte Folded Spill
	movl	%ebx, 64(%rsp)                  # 4-byte Spill
	movl	%r12d, 72(%rsp)                 # 4-byte Spill
	jmp	.LBB17_61
	.p2align	4
.LBB17_59:                              #   in Loop: Header=BB17_61 Depth=2
	movsd	%xmm0, (%rcx)
.LBB17_60:                              # %_ZNSt6vectorIdSaIdEE9push_backEOd.exit328
                                        #   in Loop: Header=BB17_61 Depth=2
	movl	24(%rsp), %eax                  # 4-byte Reload
	addq	$8, %rbp
	addq	$8, (%rsp)                      # 8-byte Folded Spill
	addq	$8, %rcx
	movq	%rcx, 80(%rsp)                  # 8-byte Spill
	decl	%eax
	je	.LBB17_95
.LBB17_61:                              #   Parent Loop BB17_58 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
.Ltmp580:                               # EH_LABEL
	movl	%eax, 24(%rsp)                  # 4-byte Spill
	movl	96(%rsp), %r15d                 # 4-byte Reload
	movl	%r15d, %edi
	movl	%r12d, %esi
	movl	%ebx, %edx
	xorl	%ecx, %ecx
	movl	$300, %r8d                      # imm = 0x12C
	callq	_Z11bench_plainiiiii
	movsd	%xmm0, 8(%rsp)                  # 8-byte Spill
.Ltmp581:                               # EH_LABEL
# %bb.62:                               #   in Loop: Header=BB17_61 Depth=2
.Ltmp583:                               # EH_LABEL
	movl	%r12d, %esi
	movq	%r13, %r12
	movq	%r14, 32(%rsp)                  # 8-byte Spill
	movl	%r15d, %edi
	movl	%ebx, %edx
	movl	$300, %ecx                      # imm = 0x12C
	callq	_Z14bench_fused_m5iiii
.Ltmp584:                               # EH_LABEL
# %bb.63:                               #   in Loop: Header=BB17_61 Depth=2
	movapd	%xmm0, %xmm1
	cmpq	16(%rsp), %rbp                  # 8-byte Folded Reload
	movsd	%xmm0, 40(%rsp)                 # 8-byte Spill
	je	.LBB17_65
# %bb.64:                               #   in Loop: Header=BB17_61 Depth=2
	movsd	8(%rsp), %xmm0                  # 8-byte Reload
                                        # xmm0 = mem[0],zero
	movsd	%xmm0, (%rbp)
	movq	%r12, %r13
	jmp	.LBB17_74
	.p2align	4
.LBB17_65:                              #   in Loop: Header=BB17_61 Depth=2
	movq	%rbp, %r13
	subq	%r12, %r13
	movabsq	$9223372036854775800, %rax      # imm = 0x7FFFFFFFFFFFFFF8
	cmpq	%rax, %r13
	je	.LBB17_379
# %bb.66:                               # %_ZNKSt6vectorIdSaIdEE12_M_check_lenEmPKc.exit.i.i
                                        #   in Loop: Header=BB17_61 Depth=2
	movq	%r13, %r14
	sarq	$3, %r14
	cmpq	$1, %r14
	adcq	%r14, %r14
	movabsq	$1152921504606846975, %rax      # imm = 0xFFFFFFFFFFFFFFF
	cmpq	%rax, %r14
	jb	.LBB17_68
# %bb.67:                               # %_ZNKSt6vectorIdSaIdEE12_M_check_lenEmPKc.exit.i.i
                                        #   in Loop: Header=BB17_61 Depth=2
	movabsq	$1152921504606846975, %r14      # imm = 0xFFFFFFFFFFFFFFF
.LBB17_68:                              # %_ZNKSt6vectorIdSaIdEE12_M_check_lenEmPKc.exit.i.i
                                        #   in Loop: Header=BB17_61 Depth=2
	leaq	(,%r14,8), %rdi
.Ltmp585:                               # EH_LABEL
	movq	%rbp, 16(%rsp)                  # 8-byte Spill
	callq	_Znwm@PLT
.Ltmp586:                               # EH_LABEL
	movsd	40(%rsp), %xmm1                 # 8-byte Reload
                                        # xmm1 = mem[0],zero
# %bb.69:                               # %.noexc308
                                        #   in Loop: Header=BB17_61 Depth=2
	movq	%rax, %rbx
	movsd	8(%rsp), %xmm0                  # 8-byte Reload
                                        # xmm0 = mem[0],zero
	movsd	%xmm0, (%rax,%r13)
	testq	%r13, %r13
	jle	.LBB17_71
# %bb.70:                               #   in Loop: Header=BB17_61 Depth=2
	movq	%rbx, %rdi
	movq	%r12, %rsi
	movq	%r13, %rdx
	callq	memcpy@PLT
	movsd	40(%rsp), %xmm1                 # 8-byte Reload
                                        # xmm1 = mem[0],zero
.LBB17_71:                              # %_ZNSt6vectorIdSaIdEE11_S_relocateEPdS2_S2_RS0_.exit.i.i
                                        #   in Loop: Header=BB17_61 Depth=2
	testq	%r12, %r12
	je	.LBB17_73
# %bb.72:                               # %_ZNSt12_Vector_baseIdSaIdEE13_M_deallocateEPdm.exit.i.i.i
                                        #   in Loop: Header=BB17_61 Depth=2
	movq	%r12, %rdi
	movq	%r13, %rsi
	callq	_ZdlPvm@PLT
	movsd	40(%rsp), %xmm1                 # 8-byte Reload
                                        # xmm1 = mem[0],zero
.LBB17_73:                              # %_ZNSt6vectorIdSaIdEE17_M_realloc_appendIJRKdEEEvDpOT_.exit.i
                                        #   in Loop: Header=BB17_61 Depth=2
	addq	%rbx, %r13
	leaq	(%rbx,%r14,8), %rax
	movq	%rax, 16(%rsp)                  # 8-byte Spill
	movq	%r13, %rbp
	movq	%rbx, %r13
.LBB17_74:                              # %_ZNSt6vectorIdSaIdEE9push_backERKd.exit
                                        #   in Loop: Header=BB17_61 Depth=2
	movq	(%rsp), %rax                    # 8-byte Reload
	cmpq	56(%rsp), %rax                  # 8-byte Folded Reload
	movq	32(%rsp), %r14                  # 8-byte Reload
	movl	72(%rsp), %r12d                 # 4-byte Reload
	je	.LBB17_76
# %bb.75:                               #   in Loop: Header=BB17_61 Depth=2
	movsd	%xmm1, (%rax)
	movl	64(%rsp), %ebx                  # 4-byte Reload
	movq	48(%rsp), %rax                  # 8-byte Reload
	jmp	.LBB17_85
	.p2align	4
.LBB17_76:                              #   in Loop: Header=BB17_61 Depth=2
	movq	%r13, %r12
	movq	%rax, %r13
	subq	88(%rsp), %r13                  # 8-byte Folded Reload
	movabsq	$9223372036854775800, %rax      # imm = 0x7FFFFFFFFFFFFFF8
	cmpq	%rax, %r13
	je	.LBB17_381
# %bb.77:                               # %_ZNKSt6vectorIdSaIdEE12_M_check_lenEmPKc.exit.i.i310
                                        #   in Loop: Header=BB17_61 Depth=2
	movq	%r13, %r14
	sarq	$3, %r14
	cmpq	$1, %r14
	adcq	%r14, %r14
	movabsq	$1152921504606846975, %rax      # imm = 0xFFFFFFFFFFFFFFF
	cmpq	%rax, %r14
	jb	.LBB17_79
# %bb.78:                               # %_ZNKSt6vectorIdSaIdEE12_M_check_lenEmPKc.exit.i.i310
                                        #   in Loop: Header=BB17_61 Depth=2
	movabsq	$1152921504606846975, %r14      # imm = 0xFFFFFFFFFFFFFFF
.LBB17_79:                              # %_ZNKSt6vectorIdSaIdEE12_M_check_lenEmPKc.exit.i.i310
                                        #   in Loop: Header=BB17_61 Depth=2
	leaq	(,%r14,8), %rdi
.Ltmp587:                               # EH_LABEL
	movq	(%rsp), %rax                    # 8-byte Reload
	movq	%rax, 56(%rsp)                  # 8-byte Spill
	callq	_Znwm@PLT
.Ltmp588:                               # EH_LABEL
	movsd	40(%rsp), %xmm1                 # 8-byte Reload
                                        # xmm1 = mem[0],zero
# %bb.80:                               # %.noexc317
                                        #   in Loop: Header=BB17_61 Depth=2
	movq	%rax, %rbx
	movsd	%xmm1, (%rax,%r13)
	testq	%r13, %r13
	movq	88(%rsp), %r15                  # 8-byte Reload
	jle	.LBB17_82
# %bb.81:                               #   in Loop: Header=BB17_61 Depth=2
	movq	%rbx, %rdi
	movq	%r15, %rsi
	movq	%r13, %rdx
	callq	memcpy@PLT
	movsd	40(%rsp), %xmm1                 # 8-byte Reload
                                        # xmm1 = mem[0],zero
.LBB17_82:                              # %_ZNSt6vectorIdSaIdEE11_S_relocateEPdS2_S2_RS0_.exit.i.i312
                                        #   in Loop: Header=BB17_61 Depth=2
	testq	%r15, %r15
	je	.LBB17_84
# %bb.83:                               # %_ZNSt12_Vector_baseIdSaIdEE13_M_deallocateEPdm.exit.i.i.i314
                                        #   in Loop: Header=BB17_61 Depth=2
	movq	%r15, %rdi
	movq	%r13, %rsi
	callq	_ZdlPvm@PLT
	movsd	40(%rsp), %xmm1                 # 8-byte Reload
                                        # xmm1 = mem[0],zero
.LBB17_84:                              # %_ZNSt6vectorIdSaIdEE17_M_realloc_appendIJRKdEEEvDpOT_.exit.i315
                                        #   in Loop: Header=BB17_61 Depth=2
	addq	%rbx, %r13
	leaq	(%rbx,%r14,8), %rax
	movq	%rax, 56(%rsp)                  # 8-byte Spill
	movq	%r13, (%rsp)                    # 8-byte Spill
	movq	%rbx, 88(%rsp)                  # 8-byte Spill
	movq	32(%rsp), %r14                  # 8-byte Reload
	movq	%r12, %r13
	movq	48(%rsp), %rax                  # 8-byte Reload
	movl	64(%rsp), %ebx                  # 4-byte Reload
	movl	72(%rsp), %r12d                 # 4-byte Reload
.LBB17_85:                              # %_ZNSt6vectorIdSaIdEE9push_backERKd.exit318
                                        #   in Loop: Header=BB17_61 Depth=2
	movsd	8(%rsp), %xmm0                  # 8-byte Reload
                                        # xmm0 = mem[0],zero
	divsd	%xmm1, %xmm0
	movq	%rax, 48(%rsp)                  # 8-byte Spill
	movq	80(%rsp), %rcx                  # 8-byte Reload
	cmpq	%rax, %rcx
	jne	.LBB17_59
# %bb.86:                               #   in Loop: Header=BB17_61 Depth=2
	movsd	%xmm0, 8(%rsp)                  # 8-byte Spill
	movq	%r13, %r12
	movq	%rcx, %r13
	subq	%r14, %r13
	movabsq	$9223372036854775800, %rax      # imm = 0x7FFFFFFFFFFFFFF8
	cmpq	%rax, %r13
	je	.LBB17_377
# %bb.87:                               # %_ZNKSt6vectorIdSaIdEE12_M_check_lenEmPKc.exit.i.i.i320
                                        #   in Loop: Header=BB17_61 Depth=2
	movq	%r13, %r14
	sarq	$3, %r14
	cmpq	$1, %r14
	adcq	%r14, %r14
	movabsq	$1152921504606846975, %rax      # imm = 0xFFFFFFFFFFFFFFF
	cmpq	%rax, %r14
	jb	.LBB17_89
# %bb.88:                               # %_ZNKSt6vectorIdSaIdEE12_M_check_lenEmPKc.exit.i.i.i320
                                        #   in Loop: Header=BB17_61 Depth=2
	movabsq	$1152921504606846975, %r14      # imm = 0xFFFFFFFFFFFFFFF
.LBB17_89:                              # %_ZNKSt6vectorIdSaIdEE12_M_check_lenEmPKc.exit.i.i.i320
                                        #   in Loop: Header=BB17_61 Depth=2
	leaq	(,%r14,8), %rdi
.Ltmp590:                               # EH_LABEL
	callq	_Znwm@PLT
.Ltmp591:                               # EH_LABEL
# %bb.90:                               # %.noexc327
                                        #   in Loop: Header=BB17_61 Depth=2
	movq	%rax, %rbx
	movsd	8(%rsp), %xmm0                  # 8-byte Reload
                                        # xmm0 = mem[0],zero
	movsd	%xmm0, (%rax,%r13)
	testq	%r13, %r13
	movq	32(%rsp), %r15                  # 8-byte Reload
	jle	.LBB17_92
# %bb.91:                               #   in Loop: Header=BB17_61 Depth=2
	movq	%rbx, %rdi
	movq	%r15, %rsi
	movq	%r13, %rdx
	callq	memcpy@PLT
.LBB17_92:                              # %_ZNSt6vectorIdSaIdEE11_S_relocateEPdS2_S2_RS0_.exit.i.i.i322
                                        #   in Loop: Header=BB17_61 Depth=2
	testq	%r15, %r15
	je	.LBB17_94
# %bb.93:                               # %_ZNSt12_Vector_baseIdSaIdEE13_M_deallocateEPdm.exit.i.i.i.i324
                                        #   in Loop: Header=BB17_61 Depth=2
	movq	%r15, %rdi
	movq	%r13, %rsi
	callq	_ZdlPvm@PLT
.LBB17_94:                              # %_ZNSt6vectorIdSaIdEE17_M_realloc_appendIJdEEEvDpOT_.exit.i.i325
                                        #   in Loop: Header=BB17_61 Depth=2
	addq	%rbx, %r13
	leaq	(%rbx,%r14,8), %rax
	movq	%rax, 48(%rsp)                  # 8-byte Spill
	movq	%r13, %rcx
	movq	%rbx, %r14
	movq	%r12, %r13
	movl	64(%rsp), %ebx                  # 4-byte Reload
	movl	72(%rsp), %r12d                 # 4-byte Reload
	jmp	.LBB17_60
	.p2align	4
.LBB17_95:                              #   in Loop: Header=BB17_58 Depth=1
	cmpq	%rbp, %r13
	je	.LBB17_98
# %bb.96:                               #   in Loop: Header=BB17_58 Depth=1
	movq	%rbp, %rbx
	subq	%r13, %rbx
	sarq	$3, %rbx
	bsrq	%rbx, %rdx
	xorl	$63, %edx
	addl	%edx, %edx
	xorq	$126, %rdx
.Ltmp593:                               # EH_LABEL
	movq	%r13, %rdi
	movq	%rbp, %rsi
	callq	_ZSt16__introsort_loopIN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEElNS0_5__ops15_Iter_less_iterEEvT_S9_T0_T1_
.Ltmp594:                               # EH_LABEL
# %bb.97:                               # %.noexc303
                                        #   in Loop: Header=BB17_58 Depth=1
.Ltmp595:                               # EH_LABEL
	movq	%r13, %rdi
	movq	%rbp, %rsi
	callq	_ZSt22__final_insertion_sortIN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEENS0_5__ops15_Iter_less_iterEEvT_S9_T0_
.Ltmp596:                               # EH_LABEL
	jmp	.LBB17_99
	.p2align	4
.LBB17_98:                              # %._crit_edge1915
                                        #   in Loop: Header=BB17_58 Depth=1
	xorl	%ebx, %ebx
.LBB17_99:                              #   in Loop: Header=BB17_58 Depth=1
	andq	$-2, %rbx
	movsd	(%r13,%rbx,4), %xmm0            # xmm0 = mem[0],zero
	movsd	%xmm0, 32(%rsp)                 # 8-byte Spill
	movq	88(%rsp), %r12                  # 8-byte Reload
	movq	(%rsp), %r15                    # 8-byte Reload
	cmpq	%r15, %r12
	je	.LBB17_102
# %bb.100:                              #   in Loop: Header=BB17_58 Depth=1
	movq	%r15, %rbx
	subq	%r12, %rbx
	sarq	$3, %rbx
	bsrq	%rbx, %rdx
	xorl	$63, %edx
	addl	%edx, %edx
	xorq	$126, %rdx
.Ltmp598:                               # EH_LABEL
	movq	%r12, %rdi
	movq	%r15, %rsi
	callq	_ZSt16__introsort_loopIN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEElNS0_5__ops15_Iter_less_iterEEvT_S9_T0_T1_
.Ltmp599:                               # EH_LABEL
# %bb.101:                              # %.noexc332
                                        #   in Loop: Header=BB17_58 Depth=1
.Ltmp600:                               # EH_LABEL
	movq	%r12, %rdi
	movq	%r15, %rsi
	callq	_ZSt22__final_insertion_sortIN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEENS0_5__ops15_Iter_less_iterEEvT_S9_T0_
.Ltmp601:                               # EH_LABEL
	jmp	.LBB17_103
	.p2align	4
.LBB17_102:                             # %._crit_edge1914
                                        #   in Loop: Header=BB17_58 Depth=1
	xorl	%ebx, %ebx
.LBB17_103:                             #   in Loop: Header=BB17_58 Depth=1
	andq	$-2, %rbx
	movq	%r12, 88(%rsp)                  # 8-byte Spill
	movsd	(%r12,%rbx,4), %xmm0            # xmm0 = mem[0],zero
	movsd	%xmm0, (%rsp)                   # 8-byte Spill
	movq	80(%rsp), %r15                  # 8-byte Reload
	cmpq	%r15, %r14
	je	.LBB17_106
# %bb.104:                              #   in Loop: Header=BB17_58 Depth=1
	movq	%r15, %rbx
	subq	%r14, %rbx
	sarq	$3, %rbx
	bsrq	%rbx, %rdx
	xorl	$63, %edx
	addl	%edx, %edx
	xorq	$126, %rdx
.Ltmp603:                               # EH_LABEL
	movq	%r14, %rdi
	movq	%r15, %rsi
	callq	_ZSt16__introsort_loopIN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEElNS0_5__ops15_Iter_less_iterEEvT_S9_T0_T1_
.Ltmp604:                               # EH_LABEL
	movl	72(%rsp), %ebp                  # 4-byte Reload
	movl	96(%rsp), %r12d                 # 4-byte Reload
# %bb.105:                              # %.noexc338
                                        #   in Loop: Header=BB17_58 Depth=1
.Ltmp605:                               # EH_LABEL
	movq	%r14, %rdi
	movq	%r15, %rsi
	callq	_ZSt22__final_insertion_sortIN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEENS0_5__ops15_Iter_less_iterEEvT_S9_T0_
.Ltmp606:                               # EH_LABEL
	jmp	.LBB17_107
	.p2align	4
.LBB17_106:                             # %._crit_edge1913
                                        #   in Loop: Header=BB17_58 Depth=1
	xorl	%ebx, %ebx
	movl	72(%rsp), %ebp                  # 4-byte Reload
	movl	96(%rsp), %r12d                 # 4-byte Reload
.LBB17_107:                             #   in Loop: Header=BB17_58 Depth=1
	andq	$-2, %rbx
	movsd	(%r14,%rbx,4), %xmm0            # xmm0 = mem[0],zero
	movsd	%xmm0, 8(%rsp)                  # 8-byte Spill
.Ltmp607:                               # EH_LABEL
	movl	$1, %edx
	movq	_ZSt4cout@GOTPCREL(%rip), %rdi
	leaq	.L.str.15(%rip), %rsi
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
.Ltmp608:                               # EH_LABEL
# %bb.108:                              # %_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc.exit342
                                        #   in Loop: Header=BB17_58 Depth=1
.Ltmp609:                               # EH_LABEL
	movq	_ZSt4cout@GOTPCREL(%rip), %rdi
	movl	%r12d, %esi
	callq	_ZNSolsEi@PLT
.Ltmp610:                               # EH_LABEL
# %bb.109:                              #   in Loop: Header=BB17_58 Depth=1
.Ltmp611:                               # EH_LABEL
	movq	%rax, %rbx
	movl	$2, %edx
	movq	%rax, %rdi
	leaq	.L.str.1(%rip), %rsi
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
.Ltmp612:                               # EH_LABEL
# %bb.110:                              # %_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc.exit344
                                        #   in Loop: Header=BB17_58 Depth=1
.Ltmp613:                               # EH_LABEL
	movq	%rbx, %rdi
	movl	%ebp, %esi
	callq	_ZNSolsEi@PLT
.Ltmp614:                               # EH_LABEL
# %bb.111:                              #   in Loop: Header=BB17_58 Depth=1
.Ltmp615:                               # EH_LABEL
	movq	%rax, %rbx
	movl	$2, %edx
	movq	%rax, %rdi
	leaq	.L.str.2(%rip), %rsi
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
.Ltmp616:                               # EH_LABEL
# %bb.112:                              # %_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc.exit346
                                        #   in Loop: Header=BB17_58 Depth=1
.Ltmp617:                               # EH_LABEL
	movq	%rbx, %rdi
	movl	64(%rsp), %esi                  # 4-byte Reload
	callq	_ZNSolsEi@PLT
.Ltmp618:                               # EH_LABEL
# %bb.113:                              #   in Loop: Header=BB17_58 Depth=1
.Ltmp619:                               # EH_LABEL
	movq	%rax, %rbx
	movl	$14, %edx
	movq	%rax, %rdi
	leaq	.L.str.22(%rip), %rsi
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
.Ltmp620:                               # EH_LABEL
# %bb.114:                              # %_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc.exit348
                                        #   in Loop: Header=BB17_58 Depth=1
.Ltmp621:                               # EH_LABEL
	movq	%rbx, %rdi
	movsd	32(%rsp), %xmm0                 # 8-byte Reload
                                        # xmm0 = mem[0],zero
	callq	_ZNSo9_M_insertIdEERSoT_@PLT
.Ltmp622:                               # EH_LABEL
# %bb.115:                              # %_ZNSolsEd.exit350
                                        #   in Loop: Header=BB17_58 Depth=1
.Ltmp623:                               # EH_LABEL
	movq	%rax, %rbx
	movl	$20, %edx
	movq	%rax, %rdi
	leaq	.L.str.23(%rip), %rsi
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
.Ltmp624:                               # EH_LABEL
# %bb.116:                              # %_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc.exit352
                                        #   in Loop: Header=BB17_58 Depth=1
.Ltmp625:                               # EH_LABEL
	movq	%rbx, %rdi
	movsd	(%rsp), %xmm0                   # 8-byte Reload
                                        # xmm0 = mem[0],zero
	callq	_ZNSo9_M_insertIdEERSoT_@PLT
.Ltmp626:                               # EH_LABEL
# %bb.117:                              # %_ZNSolsEd.exit354
                                        #   in Loop: Header=BB17_58 Depth=1
.Ltmp627:                               # EH_LABEL
	movq	%rax, %rbx
	movl	$14, %edx
	movq	%rax, %rdi
	leaq	.L.str.24(%rip), %rsi
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
.Ltmp628:                               # EH_LABEL
# %bb.118:                              # %_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc.exit356
                                        #   in Loop: Header=BB17_58 Depth=1
.Ltmp629:                               # EH_LABEL
	movq	%rbx, %rdi
	movsd	8(%rsp), %xmm0                  # 8-byte Reload
                                        # xmm0 = mem[0],zero
	callq	_ZNSo9_M_insertIdEERSoT_@PLT
.Ltmp630:                               # EH_LABEL
# %bb.119:                              # %_ZNSolsEd.exit358
                                        #   in Loop: Header=BB17_58 Depth=1
.Ltmp631:                               # EH_LABEL
	movq	%rax, %rbx
	movl	$1, %edx
	movq	%rax, %rdi
	leaq	.L.str.5(%rip), %rsi
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
.Ltmp632:                               # EH_LABEL
# %bb.120:                              # %_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc.exit360
                                        #   in Loop: Header=BB17_58 Depth=1
	movsd	8(%rsp), %xmm0                  # 8-byte Reload
                                        # xmm0 = mem[0],zero
	ucomisd	.LCPI17_0(%rip), %xmm0
	leaq	.L.str.25(%rip), %rsi
	leaq	.L.str.6(%rip), %rax
	cmovaeq	%rax, %rsi
.Ltmp633:                               # EH_LABEL
	movl	$4, %edx
	movq	%rbx, %rdi
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
.Ltmp634:                               # EH_LABEL
# %bb.121:                              # %_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc.exit363
                                        #   in Loop: Header=BB17_58 Depth=1
.Ltmp635:                               # EH_LABEL
	movl	$1, %edx
	movq	%rbx, %rdi
	leaq	.L.str.8(%rip), %rsi
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
.Ltmp636:                               # EH_LABEL
# %bb.122:                              # %_ZNSt6vectorIdSaIdEED2Ev.exit371
                                        #   in Loop: Header=BB17_58 Depth=1
	movq	88(%rsp), %rdi                  # 8-byte Reload
	movq	56(%rsp), %rsi                  # 8-byte Reload
	subq	%rdi, %rsi
	callq	_ZdlPvm@PLT
	movq	16(%rsp), %rsi                  # 8-byte Reload
	subq	%r13, %rsi
	movq	%r13, %rdi
	callq	_ZdlPvm@PLT
	movq	48(%rsp), %rsi                  # 8-byte Reload
	subq	%r14, %rsi
	movq	%r14, %rdi
	callq	_ZdlPvm@PLT
	movq	112(%rsp), %rcx                 # 8-byte Reload
	addq	$12, %rcx
	cmpq	$72, %rcx
	jne	.LBB17_58
# %bb.123:
.Ltmp638:                               # EH_LABEL
	movq	_ZSt4cout@GOTPCREL(%rip), %rdi
	leaq	.L.str.26(%rip), %rsi
	movl	$55, %edx
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
.Ltmp639:                               # EH_LABEL
# %bb.124:                              # %_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc.exit299.preheader.preheader
	xorl	%ecx, %ecx
	.p2align	4
.LBB17_125:                             # %_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc.exit299.preheader
                                        # =>This Loop Header: Depth=1
                                        #     Child Loop BB17_128 Depth 2
	movq	104(%rsp), %rax                 # 8-byte Reload
	movl	(%rax,%rcx), %edi
	movl	4(%rax,%rcx), %r14d
	movq	%rcx, 96(%rsp)                  # 8-byte Spill
	movl	8(%rax,%rcx), %ebp
	movl	$20, %eax
	movq	$0, 80(%rsp)                    # 8-byte Folded Spill
	xorl	%ebx, %ebx
	movq	$0, 56(%rsp)                    # 8-byte Folded Spill
	movq	$0, 48(%rsp)                    # 8-byte Folded Spill
	xorl	%r12d, %r12d
	movq	$0, 72(%rsp)                    # 8-byte Folded Spill
	movq	$0, 24(%rsp)                    # 8-byte Folded Spill
	xorl	%r15d, %r15d
	movq	$0, 64(%rsp)                    # 8-byte Folded Spill
	movq	%r14, 16(%rsp)                  # 8-byte Spill
	movq	%rbp, 40(%rsp)                  # 8-byte Spill
	movq	%rdi, 88(%rsp)                  # 8-byte Spill
	jmp	.LBB17_128
	.p2align	4
.LBB17_126:                             #   in Loop: Header=BB17_128 Depth=2
	movsd	%xmm0, (%r15)
.LBB17_127:                             # %_ZNSt6vectorIdSaIdEE9push_backEOd.exit415
                                        #   in Loop: Header=BB17_128 Depth=2
	movl	32(%rsp), %eax                  # 4-byte Reload
	addq	$8, %r12
	addq	$8, %rbx
	addq	$8, %r15
	decl	%eax
	movq	88(%rsp), %rdi                  # 8-byte Reload
	je	.LBB17_162
.LBB17_128:                             #   Parent Loop BB17_125 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
.Ltmp640:                               # EH_LABEL
	movl	%eax, 32(%rsp)                  # 4-byte Spill
	movl	%r14d, %esi
	movl	%ebp, %edx
	xorl	%ecx, %ecx
	movl	$300, %r8d                      # imm = 0x12C
	movq	%rdi, %r13
	callq	_Z11bench_plainiiiii
	movsd	%xmm0, 8(%rsp)                  # 8-byte Spill
.Ltmp641:                               # EH_LABEL
# %bb.129:                              #   in Loop: Header=BB17_128 Depth=2
.Ltmp643:                               # EH_LABEL
	movl	%r13d, %edi
	movl	%r14d, %esi
	movl	%ebp, %edx
	xorl	%ecx, %ecx
	movl	$300, %r8d                      # imm = 0x12C
	callq	_Z14bench_fused_m6iiiii
.Ltmp644:                               # EH_LABEL
# %bb.130:                              #   in Loop: Header=BB17_128 Depth=2
	movapd	%xmm0, %xmm1
	movq	48(%rsp), %rax                  # 8-byte Reload
	cmpq	%rax, %r12
	movsd	%xmm0, (%rsp)                   # 8-byte Spill
	je	.LBB17_132
# %bb.131:                              #   in Loop: Header=BB17_128 Depth=2
	movsd	8(%rsp), %xmm0                  # 8-byte Reload
                                        # xmm0 = mem[0],zero
	movsd	%xmm0, (%r12)
	jmp	.LBB17_141
	.p2align	4
.LBB17_132:                             #   in Loop: Header=BB17_128 Depth=2
	movq	%rax, %r12
	subq	72(%rsp), %r12                  # 8-byte Folded Reload
	movabsq	$9223372036854775800, %rax      # imm = 0x7FFFFFFFFFFFFFF8
	cmpq	%rax, %r12
	je	.LBB17_385
# %bb.133:                              # %_ZNKSt6vectorIdSaIdEE12_M_check_lenEmPKc.exit.i.i387
                                        #   in Loop: Header=BB17_128 Depth=2
	movq	%r12, %r13
	sarq	$3, %r13
	cmpq	$1, %r13
	adcq	%r13, %r13
	movabsq	$1152921504606846975, %rax      # imm = 0xFFFFFFFFFFFFFFF
	cmpq	%rax, %r13
	jb	.LBB17_135
# %bb.134:                              # %_ZNKSt6vectorIdSaIdEE12_M_check_lenEmPKc.exit.i.i387
                                        #   in Loop: Header=BB17_128 Depth=2
	movabsq	$1152921504606846975, %r13      # imm = 0xFFFFFFFFFFFFFFF
.LBB17_135:                             # %_ZNKSt6vectorIdSaIdEE12_M_check_lenEmPKc.exit.i.i387
                                        #   in Loop: Header=BB17_128 Depth=2
	leaq	(,%r13,8), %rdi
.Ltmp645:                               # EH_LABEL
	callq	_Znwm@PLT
.Ltmp646:                               # EH_LABEL
	movsd	(%rsp), %xmm1                   # 8-byte Reload
                                        # xmm1 = mem[0],zero
# %bb.136:                              # %.noexc394
                                        #   in Loop: Header=BB17_128 Depth=2
	movq	%rax, %rbp
	movsd	8(%rsp), %xmm0                  # 8-byte Reload
                                        # xmm0 = mem[0],zero
	movsd	%xmm0, (%rax,%r12)
	testq	%r12, %r12
	movq	72(%rsp), %r14                  # 8-byte Reload
	jle	.LBB17_138
# %bb.137:                              #   in Loop: Header=BB17_128 Depth=2
	movq	%rbp, %rdi
	movq	%r14, %rsi
	movq	%r12, %rdx
	callq	memcpy@PLT
	movsd	(%rsp), %xmm1                   # 8-byte Reload
                                        # xmm1 = mem[0],zero
.LBB17_138:                             # %_ZNSt6vectorIdSaIdEE11_S_relocateEPdS2_S2_RS0_.exit.i.i389
                                        #   in Loop: Header=BB17_128 Depth=2
	testq	%r14, %r14
	je	.LBB17_140
# %bb.139:                              # %_ZNSt12_Vector_baseIdSaIdEE13_M_deallocateEPdm.exit.i.i.i391
                                        #   in Loop: Header=BB17_128 Depth=2
	movq	%r14, %rdi
	movq	%r12, %rsi
	callq	_ZdlPvm@PLT
	movsd	(%rsp), %xmm1                   # 8-byte Reload
                                        # xmm1 = mem[0],zero
.LBB17_140:                             # %_ZNSt6vectorIdSaIdEE17_M_realloc_appendIJRKdEEEvDpOT_.exit.i392
                                        #   in Loop: Header=BB17_128 Depth=2
	addq	%rbp, %r12
	leaq	(,%r13,8), %rax
	addq	%rbp, %rax
	movq	%rax, 48(%rsp)                  # 8-byte Spill
	movq	%rbp, 72(%rsp)                  # 8-byte Spill
	movq	16(%rsp), %r14                  # 8-byte Reload
.LBB17_141:                             # %_ZNSt6vectorIdSaIdEE9push_backERKd.exit395
                                        #   in Loop: Header=BB17_128 Depth=2
	movq	80(%rsp), %rax                  # 8-byte Reload
	cmpq	%rax, %rbx
	je	.LBB17_143
# %bb.142:                              #   in Loop: Header=BB17_128 Depth=2
	movsd	%xmm1, (%rbx)
	jmp	.LBB17_152
	.p2align	4
.LBB17_143:                             #   in Loop: Header=BB17_128 Depth=2
	movq	%rax, %rbx
	subq	56(%rsp), %rbx                  # 8-byte Folded Reload
	movabsq	$9223372036854775800, %rax      # imm = 0x7FFFFFFFFFFFFFF8
	cmpq	%rax, %rbx
	je	.LBB17_387
# %bb.144:                              # %_ZNKSt6vectorIdSaIdEE12_M_check_lenEmPKc.exit.i.i397
                                        #   in Loop: Header=BB17_128 Depth=2
	movq	%rbx, %r13
	sarq	$3, %r13
	cmpq	$1, %r13
	adcq	%r13, %r13
	movabsq	$1152921504606846975, %rax      # imm = 0xFFFFFFFFFFFFFFF
	cmpq	%rax, %r13
	jb	.LBB17_146
# %bb.145:                              # %_ZNKSt6vectorIdSaIdEE12_M_check_lenEmPKc.exit.i.i397
                                        #   in Loop: Header=BB17_128 Depth=2
	movabsq	$1152921504606846975, %r13      # imm = 0xFFFFFFFFFFFFFFF
.LBB17_146:                             # %_ZNKSt6vectorIdSaIdEE12_M_check_lenEmPKc.exit.i.i397
                                        #   in Loop: Header=BB17_128 Depth=2
	leaq	(,%r13,8), %rdi
.Ltmp647:                               # EH_LABEL
	callq	_Znwm@PLT
.Ltmp648:                               # EH_LABEL
	movsd	(%rsp), %xmm1                   # 8-byte Reload
                                        # xmm1 = mem[0],zero
# %bb.147:                              # %.noexc404
                                        #   in Loop: Header=BB17_128 Depth=2
	movq	%rax, %rbp
	movsd	%xmm1, (%rax,%rbx)
	testq	%rbx, %rbx
	movq	56(%rsp), %r14                  # 8-byte Reload
	jle	.LBB17_149
# %bb.148:                              #   in Loop: Header=BB17_128 Depth=2
	movq	%rbp, %rdi
	movq	%r14, %rsi
	movq	%rbx, %rdx
	callq	memcpy@PLT
	movsd	(%rsp), %xmm1                   # 8-byte Reload
                                        # xmm1 = mem[0],zero
.LBB17_149:                             # %_ZNSt6vectorIdSaIdEE11_S_relocateEPdS2_S2_RS0_.exit.i.i399
                                        #   in Loop: Header=BB17_128 Depth=2
	testq	%r14, %r14
	je	.LBB17_151
# %bb.150:                              # %_ZNSt12_Vector_baseIdSaIdEE13_M_deallocateEPdm.exit.i.i.i401
                                        #   in Loop: Header=BB17_128 Depth=2
	movq	%r14, %rdi
	movq	%rbx, %rsi
	callq	_ZdlPvm@PLT
	movsd	(%rsp), %xmm1                   # 8-byte Reload
                                        # xmm1 = mem[0],zero
.LBB17_151:                             # %_ZNSt6vectorIdSaIdEE17_M_realloc_appendIJRKdEEEvDpOT_.exit.i402
                                        #   in Loop: Header=BB17_128 Depth=2
	addq	%rbp, %rbx
	leaq	(,%r13,8), %rax
	addq	%rbp, %rax
	movq	%rax, 80(%rsp)                  # 8-byte Spill
	movq	%rbp, 56(%rsp)                  # 8-byte Spill
	movq	16(%rsp), %r14                  # 8-byte Reload
.LBB17_152:                             # %_ZNSt6vectorIdSaIdEE9push_backERKd.exit405
                                        #   in Loop: Header=BB17_128 Depth=2
	movq	40(%rsp), %rbp                  # 8-byte Reload
	movsd	8(%rsp), %xmm0                  # 8-byte Reload
                                        # xmm0 = mem[0],zero
	divsd	%xmm1, %xmm0
	movq	24(%rsp), %rax                  # 8-byte Reload
	cmpq	%rax, %r15
	jne	.LBB17_126
# %bb.153:                              #   in Loop: Header=BB17_128 Depth=2
	movsd	%xmm0, 8(%rsp)                  # 8-byte Spill
	movq	%rax, %r15
	movq	64(%rsp), %r14                  # 8-byte Reload
	subq	%r14, %r15
	movabsq	$9223372036854775800, %rax      # imm = 0x7FFFFFFFFFFFFFF8
	cmpq	%rax, %r15
	je	.LBB17_383
# %bb.154:                              # %_ZNKSt6vectorIdSaIdEE12_M_check_lenEmPKc.exit.i.i.i407
                                        #   in Loop: Header=BB17_128 Depth=2
	movq	%r15, %r13
	sarq	$3, %r13
	cmpq	$1, %r13
	adcq	%r13, %r13
	movabsq	$1152921504606846975, %rax      # imm = 0xFFFFFFFFFFFFFFF
	cmpq	%rax, %r13
	jb	.LBB17_156
# %bb.155:                              # %_ZNKSt6vectorIdSaIdEE12_M_check_lenEmPKc.exit.i.i.i407
                                        #   in Loop: Header=BB17_128 Depth=2
	movabsq	$1152921504606846975, %r13      # imm = 0xFFFFFFFFFFFFFFF
.LBB17_156:                             # %_ZNKSt6vectorIdSaIdEE12_M_check_lenEmPKc.exit.i.i.i407
                                        #   in Loop: Header=BB17_128 Depth=2
	leaq	(,%r13,8), %rdi
.Ltmp650:                               # EH_LABEL
	callq	_Znwm@PLT
.Ltmp651:                               # EH_LABEL
# %bb.157:                              # %.noexc414
                                        #   in Loop: Header=BB17_128 Depth=2
	movq	%rax, %rbp
	movsd	8(%rsp), %xmm0                  # 8-byte Reload
                                        # xmm0 = mem[0],zero
	movsd	%xmm0, (%rax,%r15)
	testq	%r15, %r15
	movq	64(%rsp), %r14                  # 8-byte Reload
	jle	.LBB17_159
# %bb.158:                              #   in Loop: Header=BB17_128 Depth=2
	movq	%rbp, %rdi
	movq	%r14, %rsi
	movq	%r15, %rdx
	callq	memcpy@PLT
.LBB17_159:                             # %_ZNSt6vectorIdSaIdEE11_S_relocateEPdS2_S2_RS0_.exit.i.i.i409
                                        #   in Loop: Header=BB17_128 Depth=2
	testq	%r14, %r14
	je	.LBB17_161
# %bb.160:                              # %_ZNSt12_Vector_baseIdSaIdEE13_M_deallocateEPdm.exit.i.i.i.i411
                                        #   in Loop: Header=BB17_128 Depth=2
	movq	%r14, %rdi
	movq	%r15, %rsi
	callq	_ZdlPvm@PLT
.LBB17_161:                             # %_ZNSt6vectorIdSaIdEE17_M_realloc_appendIJdEEEvDpOT_.exit.i.i412
                                        #   in Loop: Header=BB17_128 Depth=2
	addq	%rbp, %r15
	leaq	(,%r13,8), %rax
	addq	%rbp, %rax
	movq	%rax, 24(%rsp)                  # 8-byte Spill
	movq	%rbp, 64(%rsp)                  # 8-byte Spill
	movq	16(%rsp), %r14                  # 8-byte Reload
	movq	40(%rsp), %rbp                  # 8-byte Reload
	jmp	.LBB17_127
	.p2align	4
.LBB17_162:                             #   in Loop: Header=BB17_125 Depth=1
	movq	72(%rsp), %rbp                  # 8-byte Reload
	cmpq	%r12, %rbp
	je	.LBB17_165
# %bb.163:                              #   in Loop: Header=BB17_125 Depth=1
	movq	%r12, %r14
	subq	%rbp, %r14
	sarq	$3, %r14
	bsrq	%r14, %rdx
	xorl	$63, %edx
	addl	%edx, %edx
	xorq	$126, %rdx
.Ltmp653:                               # EH_LABEL
	movq	%rbp, %rdi
	movq	%r12, %rsi
	callq	_ZSt16__introsort_loopIN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEElNS0_5__ops15_Iter_less_iterEEvT_S9_T0_T1_
.Ltmp654:                               # EH_LABEL
# %bb.164:                              # %.noexc383
                                        #   in Loop: Header=BB17_125 Depth=1
.Ltmp655:                               # EH_LABEL
	movq	%rbp, %rdi
	movq	%r12, %rsi
	callq	_ZSt22__final_insertion_sortIN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEENS0_5__ops15_Iter_less_iterEEvT_S9_T0_
.Ltmp656:                               # EH_LABEL
	jmp	.LBB17_166
	.p2align	4
.LBB17_165:                             # %._crit_edge1912
                                        #   in Loop: Header=BB17_125 Depth=1
	xorl	%r14d, %r14d
.LBB17_166:                             #   in Loop: Header=BB17_125 Depth=1
	andq	$-2, %r14
	movq	%rbp, 72(%rsp)                  # 8-byte Spill
	movsd	(%rbp,%r14,4), %xmm0            # xmm0 = mem[0],zero
	movsd	%xmm0, 8(%rsp)                  # 8-byte Spill
	movq	56(%rsp), %r12                  # 8-byte Reload
	cmpq	%rbx, %r12
	je	.LBB17_169
# %bb.167:                              #   in Loop: Header=BB17_125 Depth=1
	movq	%rbx, %r14
	subq	%r12, %r14
	sarq	$3, %r14
	bsrq	%r14, %rdx
	xorl	$63, %edx
	addl	%edx, %edx
	xorq	$126, %rdx
.Ltmp658:                               # EH_LABEL
	movq	%r12, %rdi
	movq	%rbx, %rsi
	callq	_ZSt16__introsort_loopIN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEElNS0_5__ops15_Iter_less_iterEEvT_S9_T0_T1_
.Ltmp659:                               # EH_LABEL
	movq	40(%rsp), %rbp                  # 8-byte Reload
# %bb.168:                              # %.noexc419
                                        #   in Loop: Header=BB17_125 Depth=1
.Ltmp660:                               # EH_LABEL
	movq	%r12, %rdi
	movq	%rbx, %rsi
	callq	_ZSt22__final_insertion_sortIN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEENS0_5__ops15_Iter_less_iterEEvT_S9_T0_
.Ltmp661:                               # EH_LABEL
	jmp	.LBB17_170
	.p2align	4
.LBB17_169:                             # %._crit_edge1911
                                        #   in Loop: Header=BB17_125 Depth=1
	xorl	%r14d, %r14d
	movq	40(%rsp), %rbp                  # 8-byte Reload
.LBB17_170:                             #   in Loop: Header=BB17_125 Depth=1
	andq	$-2, %r14
	movq	%r12, 56(%rsp)                  # 8-byte Spill
	movsd	(%r12,%r14,4), %xmm0            # xmm0 = mem[0],zero
	movsd	%xmm0, (%rsp)                   # 8-byte Spill
	movq	64(%rsp), %rdi                  # 8-byte Reload
	cmpq	%r15, %rdi
	je	.LBB17_173
# %bb.171:                              #   in Loop: Header=BB17_125 Depth=1
	movq	%r15, %rbx
	subq	%rdi, %rbx
	sarq	$3, %rbx
	bsrq	%rbx, %rdx
	xorl	$63, %edx
	addl	%edx, %edx
	xorq	$126, %rdx
.Ltmp663:                               # EH_LABEL
	movq	%r15, %rsi
	movq	%rdi, %r13
	callq	_ZSt16__introsort_loopIN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEElNS0_5__ops15_Iter_less_iterEEvT_S9_T0_T1_
.Ltmp664:                               # EH_LABEL
	movq	_ZSt4cout@GOTPCREL(%rip), %r14
	leaq	.L.str.27(%rip), %r12
# %bb.172:                              # %.noexc425
                                        #   in Loop: Header=BB17_125 Depth=1
.Ltmp665:                               # EH_LABEL
	movq	%r13, %rdi
	movq	%r15, %rsi
	callq	_ZSt22__final_insertion_sortIN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEENS0_5__ops15_Iter_less_iterEEvT_S9_T0_
.Ltmp666:                               # EH_LABEL
	jmp	.LBB17_174
	.p2align	4
.LBB17_173:                             # %._crit_edge1910
                                        #   in Loop: Header=BB17_125 Depth=1
	xorl	%ebx, %ebx
	movq	%rdi, %r13
	movq	_ZSt4cout@GOTPCREL(%rip), %r14
	leaq	.L.str.27(%rip), %r12
.LBB17_174:                             #   in Loop: Header=BB17_125 Depth=1
	andq	$-2, %rbx
	movq	%r13, %r15
	movsd	(%r13,%rbx,4), %xmm0            # xmm0 = mem[0],zero
	movsd	%xmm0, 32(%rsp)                 # 8-byte Spill
	movq	%rbp, %rax
	addl	$255, %ebp
	testl	%eax, %eax
	cmovnsl	%eax, %ebp
.Ltmp668:                               # EH_LABEL
	movl	$1, %edx
	movq	%r14, %rdi
	leaq	.L.str.15(%rip), %rsi
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
.Ltmp669:                               # EH_LABEL
# %bb.175:                              # %_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc.exit429
                                        #   in Loop: Header=BB17_125 Depth=1
.Ltmp670:                               # EH_LABEL
	movq	%r14, %rdi
	movq	88(%rsp), %rsi                  # 8-byte Reload
                                        # kill: def $esi killed $esi killed $rsi
	callq	_ZNSolsEi@PLT
.Ltmp671:                               # EH_LABEL
# %bb.176:                              #   in Loop: Header=BB17_125 Depth=1
.Ltmp672:                               # EH_LABEL
	movq	%rax, %rbx
	movl	$2, %edx
	movq	%rax, %rdi
	leaq	.L.str.1(%rip), %rsi
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
.Ltmp673:                               # EH_LABEL
# %bb.177:                              # %_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc.exit431
                                        #   in Loop: Header=BB17_125 Depth=1
.Ltmp674:                               # EH_LABEL
	movq	%rbx, %rdi
	movq	16(%rsp), %rsi                  # 8-byte Reload
                                        # kill: def $esi killed $esi killed $rsi
	callq	_ZNSolsEi@PLT
.Ltmp675:                               # EH_LABEL
# %bb.178:                              #   in Loop: Header=BB17_125 Depth=1
.Ltmp676:                               # EH_LABEL
	movq	%rax, %rbx
	movl	$2, %edx
	movq	%rax, %rdi
	leaq	.L.str.2(%rip), %rsi
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
.Ltmp677:                               # EH_LABEL
# %bb.179:                              # %_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc.exit433
                                        #   in Loop: Header=BB17_125 Depth=1
.Ltmp678:                               # EH_LABEL
	movq	%rbx, %rdi
	movq	40(%rsp), %rsi                  # 8-byte Reload
                                        # kill: def $esi killed $esi killed $rsi
	callq	_ZNSolsEi@PLT
.Ltmp679:                               # EH_LABEL
# %bb.180:                              #   in Loop: Header=BB17_125 Depth=1
.Ltmp680:                               # EH_LABEL
	movq	%rax, %rbx
	movl	$14, %edx
	movq	%rax, %rdi
	leaq	.L.str.22(%rip), %rsi
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
.Ltmp681:                               # EH_LABEL
# %bb.181:                              # %_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc.exit435
                                        #   in Loop: Header=BB17_125 Depth=1
.Ltmp682:                               # EH_LABEL
	movq	%rbx, %rdi
	movsd	8(%rsp), %xmm0                  # 8-byte Reload
                                        # xmm0 = mem[0],zero
	callq	_ZNSo9_M_insertIdEERSoT_@PLT
.Ltmp683:                               # EH_LABEL
# %bb.182:                              # %_ZNSolsEd.exit437
                                        #   in Loop: Header=BB17_125 Depth=1
.Ltmp684:                               # EH_LABEL
	movq	%rax, %rbx
	movl	$6, %edx
	movq	%rax, %rdi
	movq	%r12, %rsi
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
.Ltmp685:                               # EH_LABEL
# %bb.183:                              # %_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc.exit439
                                        #   in Loop: Header=BB17_125 Depth=1
.Ltmp686:                               # EH_LABEL
	xorps	%xmm0, %xmm0
	cvtsi2sdl	88(%rsp), %xmm0         # 4-byte Folded Reload
	addsd	%xmm0, %xmm0
	cvtsi2sdl	16(%rsp), %xmm1         # 4-byte Folded Reload
	mulsd	%xmm0, %xmm1
	xorps	%xmm0, %xmm0
	cvtsi2sdl	40(%rsp), %xmm0         # 4-byte Folded Reload
	mulsd	%xmm1, %xmm0
	movsd	8(%rsp), %xmm1                  # 8-byte Reload
                                        # xmm1 = mem[0],zero
	divsd	.LCPI17_1(%rip), %xmm1
	movsd	%xmm0, 64(%rsp)                 # 8-byte Spill
	movsd	%xmm1, 8(%rsp)                  # 8-byte Spill
	divsd	%xmm1, %xmm0
	divsd	.LCPI17_2(%rip), %xmm0
	movq	%rbx, %rdi
	callq	_ZNSo9_M_insertIdEERSoT_@PLT
.Ltmp687:                               # EH_LABEL
# %bb.184:                              # %_ZNSolsEd.exit441
                                        #   in Loop: Header=BB17_125 Depth=1
.Ltmp688:                               # EH_LABEL
	movq	%rax, %rbx
	movl	$4, %edx
	movq	%rax, %rdi
	leaq	.L.str.28(%rip), %rsi
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
.Ltmp689:                               # EH_LABEL
# %bb.185:                              # %_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc.exit443
                                        #   in Loop: Header=BB17_125 Depth=1
.Ltmp690:                               # EH_LABEL
	movq	16(%rsp), %rcx                  # 8-byte Reload
	movq	88(%rsp), %r14                  # 8-byte Reload
	leal	(%rcx,%r14), %eax
	imull	40(%rsp), %eax                  # 4-byte Folded Reload
                                        # kill: def $r14d killed $r14d killed $r14 def $r14
	imull	%ecx, %r14d
	leal	(%rax,%r14,4), %eax
	xorps	%xmm0, %xmm0
	cvtsi2sd	%eax, %xmm0
	divsd	8(%rsp), %xmm0                  # 8-byte Folded Reload
	divsd	.LCPI17_3(%rip), %xmm0
	movq	%rbx, %rdi
	callq	_ZNSo9_M_insertIdEERSoT_@PLT
.Ltmp691:                               # EH_LABEL
# %bb.186:                              # %_ZNSolsEd.exit445
                                        #   in Loop: Header=BB17_125 Depth=1
.Ltmp692:                               # EH_LABEL
	movq	%rax, %rbx
	movl	$22, %edx
	movq	%rax, %rdi
	leaq	.L.str.29(%rip), %rsi
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
.Ltmp693:                               # EH_LABEL
# %bb.187:                              # %_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc.exit447
                                        #   in Loop: Header=BB17_125 Depth=1
.Ltmp694:                               # EH_LABEL
	movq	%rbx, %rdi
	movsd	(%rsp), %xmm0                   # 8-byte Reload
                                        # xmm0 = mem[0],zero
	callq	_ZNSo9_M_insertIdEERSoT_@PLT
.Ltmp695:                               # EH_LABEL
# %bb.188:                              # %_ZNSolsEd.exit449
                                        #   in Loop: Header=BB17_125 Depth=1
.Ltmp696:                               # EH_LABEL
	movq	%rax, %rbx
	movl	$6, %edx
	movq	%rax, %rdi
	movq	%r12, %rsi
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
.Ltmp697:                               # EH_LABEL
# %bb.189:                              # %_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc.exit451
                                        #   in Loop: Header=BB17_125 Depth=1
.Ltmp698:                               # EH_LABEL
	movsd	(%rsp), %xmm1                   # 8-byte Reload
                                        # xmm1 = mem[0],zero
	divsd	.LCPI17_1(%rip), %xmm1
	movsd	%xmm1, (%rsp)                   # 8-byte Spill
	movsd	64(%rsp), %xmm0                 # 8-byte Reload
                                        # xmm0 = mem[0],zero
	divsd	%xmm1, %xmm0
	divsd	.LCPI17_2(%rip), %xmm0
	movq	%rbx, %rdi
	callq	_ZNSo9_M_insertIdEERSoT_@PLT
.Ltmp699:                               # EH_LABEL
# %bb.190:                              # %_ZNSolsEd.exit453
                                        #   in Loop: Header=BB17_125 Depth=1
.Ltmp700:                               # EH_LABEL
	movq	%rax, %rbx
	movl	$7, %edx
	movq	%rax, %rdi
	leaq	.L.str.30(%rip), %rsi
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
.Ltmp701:                               # EH_LABEL
# %bb.191:                              # %_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc.exit455
                                        #   in Loop: Header=BB17_125 Depth=1
.Ltmp702:                               # EH_LABEL
	movq	40(%rsp), %rax                  # 8-byte Reload
	imull	88(%rsp), %eax                  # 4-byte Folded Reload
	sarl	$8, %ebp
	movq	16(%rsp), %rcx                  # 8-byte Reload
	imull	%ebp, %ecx
	shll	$7, %ecx
	leal	(%rax,%r14,4), %eax
	addl	%ecx, %eax
	xorps	%xmm0, %xmm0
	cvtsi2sd	%eax, %xmm0
	divsd	(%rsp), %xmm0                   # 8-byte Folded Reload
	divsd	.LCPI17_3(%rip), %xmm0
	movq	%rbx, %rdi
	callq	_ZNSo9_M_insertIdEERSoT_@PLT
.Ltmp703:                               # EH_LABEL
# %bb.192:                              # %_ZNSolsEd.exit457
                                        #   in Loop: Header=BB17_125 Depth=1
.Ltmp704:                               # EH_LABEL
	movq	%rax, %rbx
	movl	$7, %edx
	movq	%rax, %rdi
	leaq	.L.str.31(%rip), %rsi
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
.Ltmp705:                               # EH_LABEL
# %bb.193:                              # %_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc.exit459
                                        #   in Loop: Header=BB17_125 Depth=1
.Ltmp706:                               # EH_LABEL
	movq	%rbx, %rdi
	movsd	32(%rsp), %xmm0                 # 8-byte Reload
                                        # xmm0 = mem[0],zero
	callq	_ZNSo9_M_insertIdEERSoT_@PLT
.Ltmp707:                               # EH_LABEL
# %bb.194:                              # %_ZNSolsEd.exit461
                                        #   in Loop: Header=BB17_125 Depth=1
.Ltmp708:                               # EH_LABEL
	movq	%rax, %rbx
	movl	$1, %edx
	movq	%rax, %rdi
	leaq	.L.str.5(%rip), %rsi
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
.Ltmp709:                               # EH_LABEL
# %bb.195:                              # %_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc.exit463
                                        #   in Loop: Header=BB17_125 Depth=1
	movsd	32(%rsp), %xmm0                 # 8-byte Reload
                                        # xmm0 = mem[0],zero
	ucomisd	.LCPI17_0(%rip), %xmm0
	leaq	.L.str.25(%rip), %rsi
	leaq	.L.str.6(%rip), %rax
	cmovaeq	%rax, %rsi
.Ltmp710:                               # EH_LABEL
	movl	$4, %edx
	movq	%rbx, %rdi
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
.Ltmp711:                               # EH_LABEL
# %bb.196:                              # %_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc.exit466
                                        #   in Loop: Header=BB17_125 Depth=1
.Ltmp712:                               # EH_LABEL
	movl	$1, %edx
	movq	%rbx, %rdi
	leaq	.L.str.8(%rip), %rsi
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
.Ltmp713:                               # EH_LABEL
# %bb.197:                              # %_ZNSt6vectorIdSaIdEED2Ev.exit474
                                        #   in Loop: Header=BB17_125 Depth=1
	movq	56(%rsp), %rdi                  # 8-byte Reload
	movq	80(%rsp), %rsi                  # 8-byte Reload
	subq	%rdi, %rsi
	callq	_ZdlPvm@PLT
	movq	72(%rsp), %rdi                  # 8-byte Reload
	movq	48(%rsp), %rsi                  # 8-byte Reload
	subq	%rdi, %rsi
	callq	_ZdlPvm@PLT
	movq	24(%rsp), %rsi                  # 8-byte Reload
	movq	%r15, %rdi
	subq	%r15, %rsi
	callq	_ZdlPvm@PLT
	movq	96(%rsp), %rcx                  # 8-byte Reload
	addq	$12, %rcx
	cmpq	$72, %rcx
	jne	.LBB17_125
# %bb.198:
.Ltmp715:                               # EH_LABEL
	movq	_ZSt4cout@GOTPCREL(%rip), %rdi
	leaq	.L.str.32(%rip), %rsi
	movl	$51, %edx
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
.Ltmp716:                               # EH_LABEL
# %bb.199:                              # %_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc.exit379.preheader.preheader
	xorl	%ecx, %ecx
	.p2align	4
.LBB17_200:                             # %_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc.exit379.preheader
                                        # =>This Loop Header: Depth=1
                                        #     Child Loop BB17_203 Depth 2
	movq	104(%rsp), %rax                 # 8-byte Reload
	movl	(%rax,%rcx), %r12d
	movl	4(%rax,%rcx), %edx
	movl	%edx, 88(%rsp)                  # 4-byte Spill
	movq	%rcx, 96(%rsp)                  # 8-byte Spill
	movl	8(%rax,%rcx), %ebx
	movl	$20, %eax
	movq	$0, 40(%rsp)                    # 8-byte Folded Spill
	xorl	%ebp, %ebp
	movq	$0, 72(%rsp)                    # 8-byte Folded Spill
	movq	$0, 24(%rsp)                    # 8-byte Folded Spill
	xorl	%r14d, %r14d
	movq	$0, 64(%rsp)                    # 8-byte Folded Spill
	movq	$0, 16(%rsp)                    # 8-byte Folded Spill
	xorl	%r13d, %r13d
	movq	$0, (%rsp)                      # 8-byte Folded Spill
	movl	%ebx, 56(%rsp)                  # 4-byte Spill
	movl	%r12d, 80(%rsp)                 # 4-byte Spill
	jmp	.LBB17_203
	.p2align	4
.LBB17_201:                             #   in Loop: Header=BB17_203 Depth=2
	movsd	%xmm0, (%r13)
.LBB17_202:                             # %_ZNSt6vectorIdSaIdEE9push_backEOd.exit518
                                        #   in Loop: Header=BB17_203 Depth=2
	movl	48(%rsp), %eax                  # 4-byte Reload
	addq	$8, %r14
	addq	$8, %rbp
	addq	$8, %r13
	decl	%eax
	je	.LBB17_237
.LBB17_203:                             #   Parent Loop BB17_200 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
.Ltmp717:                               # EH_LABEL
	movl	%eax, 48(%rsp)                  # 4-byte Spill
	movl	%r12d, %edi
	movl	88(%rsp), %r15d                 # 4-byte Reload
	movl	%r15d, %esi
	movl	%ebx, %edx
	xorl	%ecx, %ecx
	movl	$300, %r8d                      # imm = 0x12C
	callq	_Z11bench_plainiiiii
	movsd	%xmm0, 8(%rsp)                  # 8-byte Spill
.Ltmp718:                               # EH_LABEL
# %bb.204:                              #   in Loop: Header=BB17_203 Depth=2
.Ltmp720:                               # EH_LABEL
	movl	%r12d, %edi
	movl	%r15d, %esi
	movl	%ebx, %edx
	movl	$1, %ecx
	movl	$300, %r8d                      # imm = 0x12C
	callq	_Z14bench_fused_m6iiiii
.Ltmp721:                               # EH_LABEL
# %bb.205:                              #   in Loop: Header=BB17_203 Depth=2
	movapd	%xmm0, %xmm1
	movq	24(%rsp), %rax                  # 8-byte Reload
	cmpq	%rax, %r14
	movsd	%xmm0, 32(%rsp)                 # 8-byte Spill
	je	.LBB17_207
# %bb.206:                              #   in Loop: Header=BB17_203 Depth=2
	movsd	8(%rsp), %xmm0                  # 8-byte Reload
                                        # xmm0 = mem[0],zero
	movsd	%xmm0, (%r14)
	movl	80(%rsp), %r12d                 # 4-byte Reload
	movq	40(%rsp), %rax                  # 8-byte Reload
	jmp	.LBB17_216
	.p2align	4
.LBB17_207:                             #   in Loop: Header=BB17_203 Depth=2
	movq	%rax, %r14
	subq	64(%rsp), %r14                  # 8-byte Folded Reload
	movabsq	$9223372036854775800, %rax      # imm = 0x7FFFFFFFFFFFFFF8
	cmpq	%rax, %r14
	je	.LBB17_391
# %bb.208:                              # %_ZNKSt6vectorIdSaIdEE12_M_check_lenEmPKc.exit.i.i490
                                        #   in Loop: Header=BB17_203 Depth=2
	movq	%r14, %r12
	sarq	$3, %r12
	cmpq	$1, %r12
	adcq	%r12, %r12
	movabsq	$1152921504606846975, %rax      # imm = 0xFFFFFFFFFFFFFFF
	cmpq	%rax, %r12
	jb	.LBB17_210
# %bb.209:                              # %_ZNKSt6vectorIdSaIdEE12_M_check_lenEmPKc.exit.i.i490
                                        #   in Loop: Header=BB17_203 Depth=2
	movabsq	$1152921504606846975, %r12      # imm = 0xFFFFFFFFFFFFFFF
.LBB17_210:                             # %_ZNKSt6vectorIdSaIdEE12_M_check_lenEmPKc.exit.i.i490
                                        #   in Loop: Header=BB17_203 Depth=2
	leaq	(,%r12,8), %rdi
.Ltmp722:                               # EH_LABEL
	callq	_Znwm@PLT
.Ltmp723:                               # EH_LABEL
	movsd	32(%rsp), %xmm1                 # 8-byte Reload
                                        # xmm1 = mem[0],zero
# %bb.211:                              # %.noexc497
                                        #   in Loop: Header=BB17_203 Depth=2
	movq	%rax, %rbx
	movsd	8(%rsp), %xmm0                  # 8-byte Reload
                                        # xmm0 = mem[0],zero
	movsd	%xmm0, (%rax,%r14)
	testq	%r14, %r14
	movq	64(%rsp), %r15                  # 8-byte Reload
	jle	.LBB17_213
# %bb.212:                              #   in Loop: Header=BB17_203 Depth=2
	movq	%rbx, %rdi
	movq	%r15, %rsi
	movq	%r14, %rdx
	callq	memcpy@PLT
	movsd	32(%rsp), %xmm1                 # 8-byte Reload
                                        # xmm1 = mem[0],zero
.LBB17_213:                             # %_ZNSt6vectorIdSaIdEE11_S_relocateEPdS2_S2_RS0_.exit.i.i492
                                        #   in Loop: Header=BB17_203 Depth=2
	testq	%r15, %r15
	je	.LBB17_215
# %bb.214:                              # %_ZNSt12_Vector_baseIdSaIdEE13_M_deallocateEPdm.exit.i.i.i494
                                        #   in Loop: Header=BB17_203 Depth=2
	movq	%r15, %rdi
	movq	%r14, %rsi
	callq	_ZdlPvm@PLT
	movsd	32(%rsp), %xmm1                 # 8-byte Reload
                                        # xmm1 = mem[0],zero
.LBB17_215:                             # %_ZNSt6vectorIdSaIdEE17_M_realloc_appendIJRKdEEEvDpOT_.exit.i495
                                        #   in Loop: Header=BB17_203 Depth=2
	addq	%rbx, %r14
	leaq	(%rbx,%r12,8), %rax
	movq	%rax, 24(%rsp)                  # 8-byte Spill
	movq	%rbx, 64(%rsp)                  # 8-byte Spill
	movq	40(%rsp), %rax                  # 8-byte Reload
	movl	80(%rsp), %r12d                 # 4-byte Reload
.LBB17_216:                             # %_ZNSt6vectorIdSaIdEE9push_backERKd.exit498
                                        #   in Loop: Header=BB17_203 Depth=2
	cmpq	%rax, %rbp
	movq	(%rsp), %rbx                    # 8-byte Reload
	movq	%rax, 40(%rsp)                  # 8-byte Spill
	je	.LBB17_218
# %bb.217:                              #   in Loop: Header=BB17_203 Depth=2
	movq	%rbx, (%rsp)                    # 8-byte Spill
	movsd	%xmm1, (%rbp)
	movl	56(%rsp), %ebx                  # 4-byte Reload
	jmp	.LBB17_227
	.p2align	4
.LBB17_218:                             #   in Loop: Header=BB17_203 Depth=2
	movq	%rax, %rbp
	subq	72(%rsp), %rbp                  # 8-byte Folded Reload
	movabsq	$9223372036854775800, %rax      # imm = 0x7FFFFFFFFFFFFFF8
	cmpq	%rax, %rbp
	je	.LBB17_393
# %bb.219:                              # %_ZNKSt6vectorIdSaIdEE12_M_check_lenEmPKc.exit.i.i500
                                        #   in Loop: Header=BB17_203 Depth=2
	movq	%rbp, %r12
	sarq	$3, %r12
	cmpq	$1, %r12
	adcq	%r12, %r12
	movabsq	$1152921504606846975, %rax      # imm = 0xFFFFFFFFFFFFFFF
	cmpq	%rax, %r12
	jb	.LBB17_221
# %bb.220:                              # %_ZNKSt6vectorIdSaIdEE12_M_check_lenEmPKc.exit.i.i500
                                        #   in Loop: Header=BB17_203 Depth=2
	movabsq	$1152921504606846975, %r12      # imm = 0xFFFFFFFFFFFFFFF
.LBB17_221:                             # %_ZNKSt6vectorIdSaIdEE12_M_check_lenEmPKc.exit.i.i500
                                        #   in Loop: Header=BB17_203 Depth=2
	leaq	(,%r12,8), %rdi
.Ltmp724:                               # EH_LABEL
	callq	_Znwm@PLT
.Ltmp725:                               # EH_LABEL
	movsd	32(%rsp), %xmm1                 # 8-byte Reload
                                        # xmm1 = mem[0],zero
# %bb.222:                              # %.noexc507
                                        #   in Loop: Header=BB17_203 Depth=2
	movq	%rax, %rbx
	movsd	%xmm1, (%rax,%rbp)
	testq	%rbp, %rbp
	movq	72(%rsp), %r15                  # 8-byte Reload
	jle	.LBB17_224
# %bb.223:                              #   in Loop: Header=BB17_203 Depth=2
	movq	%rbx, %rdi
	movq	%r15, %rsi
	movq	%rbp, %rdx
	callq	memcpy@PLT
	movsd	32(%rsp), %xmm1                 # 8-byte Reload
                                        # xmm1 = mem[0],zero
.LBB17_224:                             # %_ZNSt6vectorIdSaIdEE11_S_relocateEPdS2_S2_RS0_.exit.i.i502
                                        #   in Loop: Header=BB17_203 Depth=2
	testq	%r15, %r15
	je	.LBB17_226
# %bb.225:                              # %_ZNSt12_Vector_baseIdSaIdEE13_M_deallocateEPdm.exit.i.i.i504
                                        #   in Loop: Header=BB17_203 Depth=2
	movq	%r15, %rdi
	movq	%rbp, %rsi
	callq	_ZdlPvm@PLT
	movsd	32(%rsp), %xmm1                 # 8-byte Reload
                                        # xmm1 = mem[0],zero
.LBB17_226:                             # %_ZNSt6vectorIdSaIdEE17_M_realloc_appendIJRKdEEEvDpOT_.exit.i505
                                        #   in Loop: Header=BB17_203 Depth=2
	addq	%rbx, %rbp
	leaq	(%rbx,%r12,8), %rax
	movq	%rax, 40(%rsp)                  # 8-byte Spill
	movq	%rbx, 72(%rsp)                  # 8-byte Spill
	movl	56(%rsp), %ebx                  # 4-byte Reload
	movl	80(%rsp), %r12d                 # 4-byte Reload
.LBB17_227:                             # %_ZNSt6vectorIdSaIdEE9push_backERKd.exit508
                                        #   in Loop: Header=BB17_203 Depth=2
	movsd	8(%rsp), %xmm0                  # 8-byte Reload
                                        # xmm0 = mem[0],zero
	divsd	%xmm1, %xmm0
	movq	16(%rsp), %rax                  # 8-byte Reload
	cmpq	%rax, %r13
	jne	.LBB17_201
# %bb.228:                              #   in Loop: Header=BB17_203 Depth=2
	movsd	%xmm0, 8(%rsp)                  # 8-byte Spill
	movq	%rax, %r13
	movq	(%rsp), %rbx                    # 8-byte Reload
	subq	%rbx, %r13
	movabsq	$9223372036854775800, %rax      # imm = 0x7FFFFFFFFFFFFFF8
	cmpq	%rax, %r13
	je	.LBB17_389
# %bb.229:                              # %_ZNKSt6vectorIdSaIdEE12_M_check_lenEmPKc.exit.i.i.i510
                                        #   in Loop: Header=BB17_203 Depth=2
	movq	%r13, %r12
	sarq	$3, %r12
	cmpq	$1, %r12
	adcq	%r12, %r12
	movabsq	$1152921504606846975, %rax      # imm = 0xFFFFFFFFFFFFFFF
	cmpq	%rax, %r12
	jb	.LBB17_231
# %bb.230:                              # %_ZNKSt6vectorIdSaIdEE12_M_check_lenEmPKc.exit.i.i.i510
                                        #   in Loop: Header=BB17_203 Depth=2
	movabsq	$1152921504606846975, %r12      # imm = 0xFFFFFFFFFFFFFFF
.LBB17_231:                             # %_ZNKSt6vectorIdSaIdEE12_M_check_lenEmPKc.exit.i.i.i510
                                        #   in Loop: Header=BB17_203 Depth=2
	leaq	(,%r12,8), %rdi
.Ltmp727:                               # EH_LABEL
	callq	_Znwm@PLT
.Ltmp728:                               # EH_LABEL
# %bb.232:                              # %.noexc517
                                        #   in Loop: Header=BB17_203 Depth=2
	movq	%rax, %rbx
	movsd	8(%rsp), %xmm0                  # 8-byte Reload
                                        # xmm0 = mem[0],zero
	movsd	%xmm0, (%rax,%r13)
	testq	%r13, %r13
	movq	(%rsp), %r15                    # 8-byte Reload
	jle	.LBB17_234
# %bb.233:                              #   in Loop: Header=BB17_203 Depth=2
	movq	%rbx, %rdi
	movq	%r15, %rsi
	movq	%r13, %rdx
	callq	memcpy@PLT
.LBB17_234:                             # %_ZNSt6vectorIdSaIdEE11_S_relocateEPdS2_S2_RS0_.exit.i.i.i512
                                        #   in Loop: Header=BB17_203 Depth=2
	testq	%r15, %r15
	je	.LBB17_236
# %bb.235:                              # %_ZNSt12_Vector_baseIdSaIdEE13_M_deallocateEPdm.exit.i.i.i.i514
                                        #   in Loop: Header=BB17_203 Depth=2
	movq	%r15, %rdi
	movq	%r13, %rsi
	callq	_ZdlPvm@PLT
.LBB17_236:                             # %_ZNSt6vectorIdSaIdEE17_M_realloc_appendIJdEEEvDpOT_.exit.i.i515
                                        #   in Loop: Header=BB17_203 Depth=2
	addq	%rbx, %r13
	leaq	(%rbx,%r12,8), %rax
	movq	%rax, 16(%rsp)                  # 8-byte Spill
	movq	%rbx, (%rsp)                    # 8-byte Spill
	movl	56(%rsp), %ebx                  # 4-byte Reload
	movl	80(%rsp), %r12d                 # 4-byte Reload
	jmp	.LBB17_202
	.p2align	4
.LBB17_237:                             #   in Loop: Header=BB17_200 Depth=1
	movq	64(%rsp), %r15                  # 8-byte Reload
	cmpq	%r14, %r15
	je	.LBB17_240
# %bb.238:                              #   in Loop: Header=BB17_200 Depth=1
	movq	%r14, %rbx
	subq	%r15, %rbx
	sarq	$3, %rbx
	bsrq	%rbx, %rdx
	xorl	$63, %edx
	addl	%edx, %edx
	xorq	$126, %rdx
.Ltmp730:                               # EH_LABEL
	movq	%r15, %rdi
	movq	%r14, %rsi
	callq	_ZSt16__introsort_loopIN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEElNS0_5__ops15_Iter_less_iterEEvT_S9_T0_T1_
.Ltmp731:                               # EH_LABEL
# %bb.239:                              # %.noexc486
                                        #   in Loop: Header=BB17_200 Depth=1
.Ltmp732:                               # EH_LABEL
	movq	%r15, %rdi
	movq	%r14, %rsi
	callq	_ZSt22__final_insertion_sortIN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEENS0_5__ops15_Iter_less_iterEEvT_S9_T0_
.Ltmp733:                               # EH_LABEL
	jmp	.LBB17_241
	.p2align	4
.LBB17_240:                             # %._crit_edge1908
                                        #   in Loop: Header=BB17_200 Depth=1
	xorl	%ebx, %ebx
.LBB17_241:                             #   in Loop: Header=BB17_200 Depth=1
	andq	$-2, %rbx
	movq	%r15, 64(%rsp)                  # 8-byte Spill
	movsd	(%r15,%rbx,4), %xmm0            # xmm0 = mem[0],zero
	movsd	%xmm0, 32(%rsp)                 # 8-byte Spill
	movq	72(%rsp), %rdi                  # 8-byte Reload
	cmpq	%rbp, %rdi
	je	.LBB17_244
# %bb.242:                              #   in Loop: Header=BB17_200 Depth=1
	movq	%rbp, %rbx
	subq	%rdi, %rbx
	sarq	$3, %rbx
	bsrq	%rbx, %rdx
	xorl	$63, %edx
	addl	%edx, %edx
	xorq	$126, %rdx
.Ltmp735:                               # EH_LABEL
	movq	%rbp, %rsi
	movq	%rdi, %r14
	callq	_ZSt16__introsort_loopIN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEElNS0_5__ops15_Iter_less_iterEEvT_S9_T0_T1_
.Ltmp736:                               # EH_LABEL
# %bb.243:                              # %.noexc522
                                        #   in Loop: Header=BB17_200 Depth=1
.Ltmp737:                               # EH_LABEL
	movq	%r14, %rdi
	movq	%rbp, %rsi
	callq	_ZSt22__final_insertion_sortIN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEENS0_5__ops15_Iter_less_iterEEvT_S9_T0_
.Ltmp738:                               # EH_LABEL
	jmp	.LBB17_245
	.p2align	4
.LBB17_244:                             # %._crit_edge1907
                                        #   in Loop: Header=BB17_200 Depth=1
	xorl	%ebx, %ebx
	movq	%rdi, %r14
.LBB17_245:                             #   in Loop: Header=BB17_200 Depth=1
	andq	$-2, %rbx
	movq	%r14, 72(%rsp)                  # 8-byte Spill
	movsd	(%r14,%rbx,4), %xmm0            # xmm0 = mem[0],zero
	movsd	%xmm0, 48(%rsp)                 # 8-byte Spill
	movq	(%rsp), %rdi                    # 8-byte Reload
	cmpq	%r13, %rdi
	je	.LBB17_248
# %bb.246:                              #   in Loop: Header=BB17_200 Depth=1
	movq	%r13, %rbx
	subq	%rdi, %rbx
	sarq	$3, %rbx
	bsrq	%rbx, %rdx
	xorl	$63, %edx
	addl	%edx, %edx
	xorq	$126, %rdx
.Ltmp740:                               # EH_LABEL
	movq	%r13, %rsi
	callq	_ZSt16__introsort_loopIN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEElNS0_5__ops15_Iter_less_iterEEvT_S9_T0_T1_
.Ltmp741:                               # EH_LABEL
	movq	_ZSt4cout@GOTPCREL(%rip), %r14
# %bb.247:                              # %.noexc528
                                        #   in Loop: Header=BB17_200 Depth=1
.Ltmp742:                               # EH_LABEL
	movq	(%rsp), %rdi                    # 8-byte Reload
	movq	%r13, %rsi
	callq	_ZSt22__final_insertion_sortIN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEENS0_5__ops15_Iter_less_iterEEvT_S9_T0_
.Ltmp743:                               # EH_LABEL
	jmp	.LBB17_249
	.p2align	4
.LBB17_248:                             # %._crit_edge1906
                                        #   in Loop: Header=BB17_200 Depth=1
	xorl	%ebx, %ebx
	movq	_ZSt4cout@GOTPCREL(%rip), %r14
.LBB17_249:                             #   in Loop: Header=BB17_200 Depth=1
	andq	$-2, %rbx
	movq	(%rsp), %rax                    # 8-byte Reload
	movsd	(%rax,%rbx,4), %xmm0            # xmm0 = mem[0],zero
	movsd	%xmm0, 8(%rsp)                  # 8-byte Spill
.Ltmp744:                               # EH_LABEL
	movl	$1, %edx
	movq	%r14, %rdi
	leaq	.L.str.15(%rip), %rsi
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
.Ltmp745:                               # EH_LABEL
# %bb.250:                              # %_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc.exit532
                                        #   in Loop: Header=BB17_200 Depth=1
.Ltmp746:                               # EH_LABEL
	movq	%r14, %rdi
	movl	%r12d, %esi
	callq	_ZNSolsEi@PLT
.Ltmp747:                               # EH_LABEL
# %bb.251:                              #   in Loop: Header=BB17_200 Depth=1
.Ltmp748:                               # EH_LABEL
	movq	%rax, %rbx
	movl	$2, %edx
	movq	%rax, %rdi
	leaq	.L.str.1(%rip), %rsi
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
.Ltmp749:                               # EH_LABEL
# %bb.252:                              # %_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc.exit534
                                        #   in Loop: Header=BB17_200 Depth=1
.Ltmp750:                               # EH_LABEL
	movq	%rbx, %rdi
	movl	88(%rsp), %esi                  # 4-byte Reload
	callq	_ZNSolsEi@PLT
.Ltmp751:                               # EH_LABEL
# %bb.253:                              #   in Loop: Header=BB17_200 Depth=1
.Ltmp752:                               # EH_LABEL
	movq	%rax, %rbx
	movl	$2, %edx
	movq	%rax, %rdi
	leaq	.L.str.2(%rip), %rsi
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
.Ltmp753:                               # EH_LABEL
# %bb.254:                              # %_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc.exit536
                                        #   in Loop: Header=BB17_200 Depth=1
.Ltmp754:                               # EH_LABEL
	movq	%rbx, %rdi
	movl	56(%rsp), %esi                  # 4-byte Reload
	callq	_ZNSolsEi@PLT
.Ltmp755:                               # EH_LABEL
# %bb.255:                              #   in Loop: Header=BB17_200 Depth=1
.Ltmp756:                               # EH_LABEL
	movq	%rax, %rbx
	movl	$14, %edx
	movq	%rax, %rdi
	leaq	.L.str.22(%rip), %rsi
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
.Ltmp757:                               # EH_LABEL
# %bb.256:                              # %_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc.exit538
                                        #   in Loop: Header=BB17_200 Depth=1
.Ltmp758:                               # EH_LABEL
	movq	%rbx, %rdi
	movsd	32(%rsp), %xmm0                 # 8-byte Reload
                                        # xmm0 = mem[0],zero
	callq	_ZNSo9_M_insertIdEERSoT_@PLT
.Ltmp759:                               # EH_LABEL
# %bb.257:                              # %_ZNSolsEd.exit540
                                        #   in Loop: Header=BB17_200 Depth=1
.Ltmp760:                               # EH_LABEL
	movq	%rax, %rbx
	movl	$14, %edx
	movq	%rax, %rdi
	leaq	.L.str.33(%rip), %rsi
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
.Ltmp761:                               # EH_LABEL
# %bb.258:                              # %_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc.exit542
                                        #   in Loop: Header=BB17_200 Depth=1
.Ltmp762:                               # EH_LABEL
	movq	%rbx, %rdi
	movsd	48(%rsp), %xmm0                 # 8-byte Reload
                                        # xmm0 = mem[0],zero
	callq	_ZNSo9_M_insertIdEERSoT_@PLT
.Ltmp763:                               # EH_LABEL
# %bb.259:                              # %_ZNSolsEd.exit544
                                        #   in Loop: Header=BB17_200 Depth=1
.Ltmp764:                               # EH_LABEL
	movq	%rax, %rbx
	movl	$7, %edx
	movq	%rax, %rdi
	leaq	.L.str.31(%rip), %rsi
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
.Ltmp765:                               # EH_LABEL
# %bb.260:                              # %_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc.exit546
                                        #   in Loop: Header=BB17_200 Depth=1
.Ltmp766:                               # EH_LABEL
	movq	%rbx, %rdi
	movsd	8(%rsp), %xmm0                  # 8-byte Reload
                                        # xmm0 = mem[0],zero
	callq	_ZNSo9_M_insertIdEERSoT_@PLT
.Ltmp767:                               # EH_LABEL
# %bb.261:                              # %_ZNSolsEd.exit548
                                        #   in Loop: Header=BB17_200 Depth=1
.Ltmp768:                               # EH_LABEL
	movq	%rax, %rbx
	movl	$1, %edx
	movq	%rax, %rdi
	leaq	.L.str.5(%rip), %rsi
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
.Ltmp769:                               # EH_LABEL
# %bb.262:                              # %_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc.exit550
                                        #   in Loop: Header=BB17_200 Depth=1
	movsd	8(%rsp), %xmm0                  # 8-byte Reload
                                        # xmm0 = mem[0],zero
	ucomisd	.LCPI17_0(%rip), %xmm0
	leaq	.L.str.25(%rip), %rsi
	leaq	.L.str.6(%rip), %rax
	cmovaeq	%rax, %rsi
.Ltmp770:                               # EH_LABEL
	movl	$4, %edx
	movq	%rbx, %rdi
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
.Ltmp771:                               # EH_LABEL
# %bb.263:                              # %_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc.exit553
                                        #   in Loop: Header=BB17_200 Depth=1
.Ltmp772:                               # EH_LABEL
	movl	$1, %edx
	movq	%rbx, %rdi
	leaq	.L.str.8(%rip), %rsi
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
.Ltmp773:                               # EH_LABEL
# %bb.264:                              # %_ZNSt6vectorIdSaIdEED2Ev.exit561
                                        #   in Loop: Header=BB17_200 Depth=1
	movq	72(%rsp), %rdi                  # 8-byte Reload
	movq	40(%rsp), %rsi                  # 8-byte Reload
	subq	%rdi, %rsi
	callq	_ZdlPvm@PLT
	movq	64(%rsp), %rdi                  # 8-byte Reload
	movq	24(%rsp), %rsi                  # 8-byte Reload
	subq	%rdi, %rsi
	callq	_ZdlPvm@PLT
	movq	16(%rsp), %rsi                  # 8-byte Reload
	movq	(%rsp), %rdi                    # 8-byte Reload
	subq	%rdi, %rsi
	callq	_ZdlPvm@PLT
	movq	96(%rsp), %rcx                  # 8-byte Reload
	addq	$12, %rcx
	cmpq	$72, %rcx
	jne	.LBB17_200
# %bb.265:
.Ltmp775:                               # EH_LABEL
	movq	_ZSt4cout@GOTPCREL(%rip), %rdi
	leaq	.L.str.34(%rip), %rsi
	movl	$43, %edx
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
.Ltmp776:                               # EH_LABEL
# %bb.266:                              # %_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc.exit482.preheader.preheader
	xorl	%ecx, %ecx
	jmp	.LBB17_268
	.p2align	4
.LBB17_267:                             # %_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc.exit482
                                        #   in Loop: Header=BB17_268 Depth=1
	addq	$12, %rcx
	cmpq	$72, %rcx
	je	.LBB17_334
.LBB17_268:                             # %_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc.exit482.preheader
                                        # =>This Loop Header: Depth=1
                                        #     Child Loop BB17_272 Depth 2
	movq	104(%rsp), %rax                 # 8-byte Reload
	movl	8(%rax,%rcx), %eax
	testb	$63, %al
	jne	.LBB17_267
# %bb.269:                              # %.preheader.preheader
                                        #   in Loop: Header=BB17_268 Depth=1
	movl	%eax, 88(%rsp)                  # 4-byte Spill
	movq	104(%rsp), %rax                 # 8-byte Reload
	movl	(%rax,%rcx), %edx
	movl	%edx, 72(%rsp)                  # 4-byte Spill
	movq	%rcx, 96(%rsp)                  # 8-byte Spill
	movl	4(%rax,%rcx), %eax
	movl	%eax, 64(%rsp)                  # 4-byte Spill
	movl	$20, %eax
	movq	$0, 80(%rsp)                    # 8-byte Folded Spill
	xorl	%ebp, %ebp
	movq	$0, 16(%rsp)                    # 8-byte Folded Spill
	movq	$0, 48(%rsp)                    # 8-byte Folded Spill
	xorl	%r14d, %r14d
	movq	$0, 56(%rsp)                    # 8-byte Folded Spill
	movq	$0, 32(%rsp)                    # 8-byte Folded Spill
	xorl	%r13d, %r13d
	movq	$0, 24(%rsp)                    # 8-byte Folded Spill
	jmp	.LBB17_272
	.p2align	4
.LBB17_270:                             #   in Loop: Header=BB17_272 Depth=2
	movq	%rcx, 32(%rsp)                  # 8-byte Spill
	movsd	%xmm0, (%r13)
.LBB17_271:                             # %_ZNSt6vectorIdSaIdEE9push_backEOd.exit605
                                        #   in Loop: Header=BB17_272 Depth=2
	movl	40(%rsp), %eax                  # 4-byte Reload
	addq	$8, %r14
	addq	$8, %rbp
	addq	$8, %r13
	decl	%eax
	je	.LBB17_306
.LBB17_272:                             # %.preheader
                                        #   Parent Loop BB17_268 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
.Ltmp777:                               # EH_LABEL
	movl	%eax, 40(%rsp)                  # 4-byte Spill
	movl	72(%rsp), %r12d                 # 4-byte Reload
	movl	%r12d, %edi
	movl	64(%rsp), %r15d                 # 4-byte Reload
	movl	%r15d, %esi
	movl	88(%rsp), %ebx                  # 4-byte Reload
	movl	%ebx, %edx
	xorl	%ecx, %ecx
	movl	$300, %r8d                      # imm = 0x12C
	callq	_Z11bench_plainiiiii
	movsd	%xmm0, 8(%rsp)                  # 8-byte Spill
.Ltmp778:                               # EH_LABEL
# %bb.273:                              #   in Loop: Header=BB17_272 Depth=2
.Ltmp780:                               # EH_LABEL
	movl	%r12d, %edi
	movl	%r15d, %esi
	movl	%ebx, %edx
	movl	$2, %ecx
	movl	$300, %r8d                      # imm = 0x12C
	callq	_Z14bench_fused_m6iiiii
.Ltmp781:                               # EH_LABEL
# %bb.274:                              #   in Loop: Header=BB17_272 Depth=2
	movapd	%xmm0, %xmm1
	movq	48(%rsp), %rax                  # 8-byte Reload
	cmpq	%rax, %r14
	movsd	%xmm0, (%rsp)                   # 8-byte Spill
	je	.LBB17_276
# %bb.275:                              #   in Loop: Header=BB17_272 Depth=2
	movsd	8(%rsp), %xmm0                  # 8-byte Reload
                                        # xmm0 = mem[0],zero
	movsd	%xmm0, (%r14)
	jmp	.LBB17_285
	.p2align	4
.LBB17_276:                             #   in Loop: Header=BB17_272 Depth=2
	movq	%rax, %r14
	subq	56(%rsp), %r14                  # 8-byte Folded Reload
	movabsq	$9223372036854775800, %rax      # imm = 0x7FFFFFFFFFFFFFF8
	cmpq	%rax, %r14
	je	.LBB17_397
# %bb.277:                              # %_ZNKSt6vectorIdSaIdEE12_M_check_lenEmPKc.exit.i.i577
                                        #   in Loop: Header=BB17_272 Depth=2
	movq	%r14, %r12
	sarq	$3, %r12
	cmpq	$1, %r12
	adcq	%r12, %r12
	movabsq	$1152921504606846975, %rax      # imm = 0xFFFFFFFFFFFFFFF
	cmpq	%rax, %r12
	jb	.LBB17_279
# %bb.278:                              # %_ZNKSt6vectorIdSaIdEE12_M_check_lenEmPKc.exit.i.i577
                                        #   in Loop: Header=BB17_272 Depth=2
	movabsq	$1152921504606846975, %r12      # imm = 0xFFFFFFFFFFFFFFF
.LBB17_279:                             # %_ZNKSt6vectorIdSaIdEE12_M_check_lenEmPKc.exit.i.i577
                                        #   in Loop: Header=BB17_272 Depth=2
	leaq	(,%r12,8), %rdi
.Ltmp782:                               # EH_LABEL
	callq	_Znwm@PLT
.Ltmp783:                               # EH_LABEL
	movsd	(%rsp), %xmm1                   # 8-byte Reload
                                        # xmm1 = mem[0],zero
# %bb.280:                              # %.noexc584
                                        #   in Loop: Header=BB17_272 Depth=2
	movq	%rax, %r15
	movsd	8(%rsp), %xmm0                  # 8-byte Reload
                                        # xmm0 = mem[0],zero
	movsd	%xmm0, (%rax,%r14)
	testq	%r14, %r14
	movq	56(%rsp), %rbx                  # 8-byte Reload
	jle	.LBB17_282
# %bb.281:                              #   in Loop: Header=BB17_272 Depth=2
	movq	%r15, %rdi
	movq	%rbx, %rsi
	movq	%r14, %rdx
	callq	memcpy@PLT
	movsd	(%rsp), %xmm1                   # 8-byte Reload
                                        # xmm1 = mem[0],zero
.LBB17_282:                             # %_ZNSt6vectorIdSaIdEE11_S_relocateEPdS2_S2_RS0_.exit.i.i579
                                        #   in Loop: Header=BB17_272 Depth=2
	testq	%rbx, %rbx
	je	.LBB17_284
# %bb.283:                              # %_ZNSt12_Vector_baseIdSaIdEE13_M_deallocateEPdm.exit.i.i.i581
                                        #   in Loop: Header=BB17_272 Depth=2
	movq	%rbx, %rdi
	movq	%r14, %rsi
	callq	_ZdlPvm@PLT
	movsd	(%rsp), %xmm1                   # 8-byte Reload
                                        # xmm1 = mem[0],zero
.LBB17_284:                             # %_ZNSt6vectorIdSaIdEE17_M_realloc_appendIJRKdEEEvDpOT_.exit.i582
                                        #   in Loop: Header=BB17_272 Depth=2
	addq	%r15, %r14
	leaq	(%r15,%r12,8), %rax
	movq	%rax, 48(%rsp)                  # 8-byte Spill
	movq	%r15, 56(%rsp)                  # 8-byte Spill
.LBB17_285:                             # %_ZNSt6vectorIdSaIdEE9push_backERKd.exit585
                                        #   in Loop: Header=BB17_272 Depth=2
	movq	80(%rsp), %rax                  # 8-byte Reload
	cmpq	%rax, %rbp
	je	.LBB17_287
# %bb.286:                              #   in Loop: Header=BB17_272 Depth=2
	movsd	%xmm1, (%rbp)
	jmp	.LBB17_296
	.p2align	4
.LBB17_287:                             #   in Loop: Header=BB17_272 Depth=2
	movq	%rax, %rbp
	subq	16(%rsp), %rbp                  # 8-byte Folded Reload
	movabsq	$9223372036854775800, %rax      # imm = 0x7FFFFFFFFFFFFFF8
	cmpq	%rax, %rbp
	je	.LBB17_399
# %bb.288:                              # %_ZNKSt6vectorIdSaIdEE12_M_check_lenEmPKc.exit.i.i587
                                        #   in Loop: Header=BB17_272 Depth=2
	movq	%rbp, %r12
	sarq	$3, %r12
	cmpq	$1, %r12
	adcq	%r12, %r12
	movabsq	$1152921504606846975, %rax      # imm = 0xFFFFFFFFFFFFFFF
	cmpq	%rax, %r12
	jb	.LBB17_290
# %bb.289:                              # %_ZNKSt6vectorIdSaIdEE12_M_check_lenEmPKc.exit.i.i587
                                        #   in Loop: Header=BB17_272 Depth=2
	movabsq	$1152921504606846975, %r12      # imm = 0xFFFFFFFFFFFFFFF
.LBB17_290:                             # %_ZNKSt6vectorIdSaIdEE12_M_check_lenEmPKc.exit.i.i587
                                        #   in Loop: Header=BB17_272 Depth=2
	leaq	(,%r12,8), %rdi
.Ltmp784:                               # EH_LABEL
	callq	_Znwm@PLT
.Ltmp785:                               # EH_LABEL
	movsd	(%rsp), %xmm1                   # 8-byte Reload
                                        # xmm1 = mem[0],zero
# %bb.291:                              # %.noexc594
                                        #   in Loop: Header=BB17_272 Depth=2
	movq	%rax, %r15
	movsd	%xmm1, (%rax,%rbp)
	testq	%rbp, %rbp
	movq	16(%rsp), %rbx                  # 8-byte Reload
	jle	.LBB17_293
# %bb.292:                              #   in Loop: Header=BB17_272 Depth=2
	movq	%r15, %rdi
	movq	%rbx, %rsi
	movq	%rbp, %rdx
	callq	memcpy@PLT
	movsd	(%rsp), %xmm1                   # 8-byte Reload
                                        # xmm1 = mem[0],zero
.LBB17_293:                             # %_ZNSt6vectorIdSaIdEE11_S_relocateEPdS2_S2_RS0_.exit.i.i589
                                        #   in Loop: Header=BB17_272 Depth=2
	testq	%rbx, %rbx
	je	.LBB17_295
# %bb.294:                              # %_ZNSt12_Vector_baseIdSaIdEE13_M_deallocateEPdm.exit.i.i.i591
                                        #   in Loop: Header=BB17_272 Depth=2
	movq	%rbx, %rdi
	movq	%rbp, %rsi
	callq	_ZdlPvm@PLT
	movsd	(%rsp), %xmm1                   # 8-byte Reload
                                        # xmm1 = mem[0],zero
.LBB17_295:                             # %_ZNSt6vectorIdSaIdEE17_M_realloc_appendIJRKdEEEvDpOT_.exit.i592
                                        #   in Loop: Header=BB17_272 Depth=2
	addq	%r15, %rbp
	leaq	(%r15,%r12,8), %rax
	movq	%rax, 80(%rsp)                  # 8-byte Spill
	movq	%r15, 16(%rsp)                  # 8-byte Spill
.LBB17_296:                             # %_ZNSt6vectorIdSaIdEE9push_backERKd.exit595
                                        #   in Loop: Header=BB17_272 Depth=2
	movq	32(%rsp), %rcx                  # 8-byte Reload
	movsd	8(%rsp), %xmm0                  # 8-byte Reload
                                        # xmm0 = mem[0],zero
	divsd	%xmm1, %xmm0
	cmpq	%rcx, %r13
	jne	.LBB17_270
# %bb.297:                              #   in Loop: Header=BB17_272 Depth=2
	movsd	%xmm0, 8(%rsp)                  # 8-byte Spill
	movq	%rcx, %r13
	subq	24(%rsp), %r13                  # 8-byte Folded Reload
	movabsq	$9223372036854775800, %rax      # imm = 0x7FFFFFFFFFFFFFF8
	cmpq	%rax, %r13
	je	.LBB17_395
# %bb.298:                              # %_ZNKSt6vectorIdSaIdEE12_M_check_lenEmPKc.exit.i.i.i597
                                        #   in Loop: Header=BB17_272 Depth=2
	movq	%r13, %r12
	sarq	$3, %r12
	cmpq	$1, %r12
	adcq	%r12, %r12
	movabsq	$1152921504606846975, %rax      # imm = 0xFFFFFFFFFFFFFFF
	cmpq	%rax, %r12
	jb	.LBB17_300
# %bb.299:                              # %_ZNKSt6vectorIdSaIdEE12_M_check_lenEmPKc.exit.i.i.i597
                                        #   in Loop: Header=BB17_272 Depth=2
	movabsq	$1152921504606846975, %r12      # imm = 0xFFFFFFFFFFFFFFF
.LBB17_300:                             # %_ZNKSt6vectorIdSaIdEE12_M_check_lenEmPKc.exit.i.i.i597
                                        #   in Loop: Header=BB17_272 Depth=2
	leaq	(,%r12,8), %rdi
.Ltmp787:                               # EH_LABEL
	callq	_Znwm@PLT
.Ltmp788:                               # EH_LABEL
# %bb.301:                              # %.noexc604
                                        #   in Loop: Header=BB17_272 Depth=2
	movq	%rax, %r15
	movsd	8(%rsp), %xmm0                  # 8-byte Reload
                                        # xmm0 = mem[0],zero
	movsd	%xmm0, (%rax,%r13)
	testq	%r13, %r13
	movq	24(%rsp), %rbx                  # 8-byte Reload
	jle	.LBB17_303
# %bb.302:                              #   in Loop: Header=BB17_272 Depth=2
	movq	%r15, %rdi
	movq	%rbx, %rsi
	movq	%r13, %rdx
	callq	memcpy@PLT
.LBB17_303:                             # %_ZNSt6vectorIdSaIdEE11_S_relocateEPdS2_S2_RS0_.exit.i.i.i599
                                        #   in Loop: Header=BB17_272 Depth=2
	testq	%rbx, %rbx
	je	.LBB17_305
# %bb.304:                              # %_ZNSt12_Vector_baseIdSaIdEE13_M_deallocateEPdm.exit.i.i.i.i601
                                        #   in Loop: Header=BB17_272 Depth=2
	movq	%rbx, %rdi
	movq	%r13, %rsi
	callq	_ZdlPvm@PLT
.LBB17_305:                             # %_ZNSt6vectorIdSaIdEE17_M_realloc_appendIJdEEEvDpOT_.exit.i.i602
                                        #   in Loop: Header=BB17_272 Depth=2
	addq	%r15, %r13
	leaq	(%r15,%r12,8), %rax
	movq	%rax, 32(%rsp)                  # 8-byte Spill
	movq	%r15, 24(%rsp)                  # 8-byte Spill
	jmp	.LBB17_271
	.p2align	4
.LBB17_306:                             #   in Loop: Header=BB17_268 Depth=1
	movq	56(%rsp), %r15                  # 8-byte Reload
	cmpq	%r14, %r15
	je	.LBB17_309
# %bb.307:                              #   in Loop: Header=BB17_268 Depth=1
	movq	%r14, %rbx
	subq	%r15, %rbx
	sarq	$3, %rbx
	bsrq	%rbx, %rdx
	xorl	$63, %edx
	addl	%edx, %edx
	xorq	$126, %rdx
.Ltmp790:                               # EH_LABEL
	movq	%r15, %rdi
	movq	%r14, %rsi
	callq	_ZSt16__introsort_loopIN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEElNS0_5__ops15_Iter_less_iterEEvT_S9_T0_T1_
.Ltmp791:                               # EH_LABEL
# %bb.308:                              # %.noexc573
                                        #   in Loop: Header=BB17_268 Depth=1
.Ltmp792:                               # EH_LABEL
	movq	%r15, %rdi
	movq	%r14, %rsi
	callq	_ZSt22__final_insertion_sortIN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEENS0_5__ops15_Iter_less_iterEEvT_S9_T0_
.Ltmp793:                               # EH_LABEL
	jmp	.LBB17_310
.LBB17_309:                             # %._crit_edge1904
                                        #   in Loop: Header=BB17_268 Depth=1
	xorl	%ebx, %ebx
.LBB17_310:                             #   in Loop: Header=BB17_268 Depth=1
	andq	$-2, %rbx
	movq	%r15, 56(%rsp)                  # 8-byte Spill
	movsd	(%r15,%rbx,4), %xmm0            # xmm0 = mem[0],zero
	movsd	%xmm0, (%rsp)                   # 8-byte Spill
	movq	16(%rsp), %r14                  # 8-byte Reload
	cmpq	%rbp, %r14
	je	.LBB17_313
# %bb.311:                              #   in Loop: Header=BB17_268 Depth=1
	movq	%rbp, %rbx
	subq	%r14, %rbx
	sarq	$3, %rbx
	bsrq	%rbx, %rdx
	xorl	$63, %edx
	addl	%edx, %edx
	xorq	$126, %rdx
.Ltmp795:                               # EH_LABEL
	movq	%r14, %rdi
	movq	%rbp, %rsi
	callq	_ZSt16__introsort_loopIN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEElNS0_5__ops15_Iter_less_iterEEvT_S9_T0_T1_
.Ltmp796:                               # EH_LABEL
# %bb.312:                              # %.noexc609
                                        #   in Loop: Header=BB17_268 Depth=1
.Ltmp797:                               # EH_LABEL
	movq	%r14, %rdi
	movq	%rbp, %rsi
	callq	_ZSt22__final_insertion_sortIN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEENS0_5__ops15_Iter_less_iterEEvT_S9_T0_
.Ltmp798:                               # EH_LABEL
	jmp	.LBB17_314
.LBB17_313:                             # %._crit_edge1903
                                        #   in Loop: Header=BB17_268 Depth=1
	xorl	%ebx, %ebx
.LBB17_314:                             #   in Loop: Header=BB17_268 Depth=1
	andq	$-2, %rbx
	movq	%r14, 16(%rsp)                  # 8-byte Spill
	movsd	(%r14,%rbx,4), %xmm0            # xmm0 = mem[0],zero
	movsd	%xmm0, 40(%rsp)                 # 8-byte Spill
	movq	24(%rsp), %rdi                  # 8-byte Reload
	cmpq	%r13, %rdi
	je	.LBB17_317
# %bb.315:                              #   in Loop: Header=BB17_268 Depth=1
	movq	%r13, %rbx
	subq	%rdi, %rbx
	sarq	$3, %rbx
	bsrq	%rbx, %rdx
	xorl	$63, %edx
	addl	%edx, %edx
	xorq	$126, %rdx
.Ltmp800:                               # EH_LABEL
	movq	%r13, %rsi
	callq	_ZSt16__introsort_loopIN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEElNS0_5__ops15_Iter_less_iterEEvT_S9_T0_T1_
.Ltmp801:                               # EH_LABEL
	movl	72(%rsp), %ebp                  # 4-byte Reload
# %bb.316:                              # %.noexc615
                                        #   in Loop: Header=BB17_268 Depth=1
.Ltmp802:                               # EH_LABEL
	movq	24(%rsp), %rdi                  # 8-byte Reload
	movq	%r13, %rsi
	callq	_ZSt22__final_insertion_sortIN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEENS0_5__ops15_Iter_less_iterEEvT_S9_T0_
.Ltmp803:                               # EH_LABEL
	jmp	.LBB17_318
.LBB17_317:                             # %._crit_edge
                                        #   in Loop: Header=BB17_268 Depth=1
	xorl	%ebx, %ebx
	movl	72(%rsp), %ebp                  # 4-byte Reload
.LBB17_318:                             #   in Loop: Header=BB17_268 Depth=1
	andq	$-2, %rbx
	movq	24(%rsp), %rax                  # 8-byte Reload
	movsd	(%rax,%rbx,4), %xmm0            # xmm0 = mem[0],zero
	movsd	%xmm0, 8(%rsp)                  # 8-byte Spill
.Ltmp804:                               # EH_LABEL
	movl	$1, %edx
	movq	_ZSt4cout@GOTPCREL(%rip), %rdi
	leaq	.L.str.15(%rip), %rsi
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
.Ltmp805:                               # EH_LABEL
# %bb.319:                              # %_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc.exit619
                                        #   in Loop: Header=BB17_268 Depth=1
.Ltmp806:                               # EH_LABEL
	movq	_ZSt4cout@GOTPCREL(%rip), %rdi
	movl	%ebp, %esi
	callq	_ZNSolsEi@PLT
.Ltmp807:                               # EH_LABEL
# %bb.320:                              #   in Loop: Header=BB17_268 Depth=1
.Ltmp808:                               # EH_LABEL
	movq	%rax, %r14
	movl	$2, %edx
	movq	%rax, %rdi
	leaq	.L.str.1(%rip), %rsi
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
.Ltmp809:                               # EH_LABEL
# %bb.321:                              # %_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc.exit621
                                        #   in Loop: Header=BB17_268 Depth=1
.Ltmp810:                               # EH_LABEL
	movq	%r14, %rdi
	movl	64(%rsp), %esi                  # 4-byte Reload
	callq	_ZNSolsEi@PLT
.Ltmp811:                               # EH_LABEL
# %bb.322:                              #   in Loop: Header=BB17_268 Depth=1
.Ltmp812:                               # EH_LABEL
	movq	%rax, %r14
	movl	$2, %edx
	movq	%rax, %rdi
	leaq	.L.str.2(%rip), %rsi
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
.Ltmp813:                               # EH_LABEL
# %bb.323:                              # %_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc.exit623
                                        #   in Loop: Header=BB17_268 Depth=1
.Ltmp814:                               # EH_LABEL
	movq	%r14, %rdi
	movl	88(%rsp), %esi                  # 4-byte Reload
	callq	_ZNSolsEi@PLT
.Ltmp815:                               # EH_LABEL
# %bb.324:                              #   in Loop: Header=BB17_268 Depth=1
.Ltmp816:                               # EH_LABEL
	movq	%rax, %rbx
	movl	$14, %edx
	movq	%rax, %rdi
	leaq	.L.str.22(%rip), %rsi
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
.Ltmp817:                               # EH_LABEL
# %bb.325:                              # %_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc.exit625
                                        #   in Loop: Header=BB17_268 Depth=1
.Ltmp818:                               # EH_LABEL
	movq	%rbx, %rdi
	movsd	(%rsp), %xmm0                   # 8-byte Reload
                                        # xmm0 = mem[0],zero
	callq	_ZNSo9_M_insertIdEERSoT_@PLT
.Ltmp819:                               # EH_LABEL
# %bb.326:                              # %_ZNSolsEd.exit627
                                        #   in Loop: Header=BB17_268 Depth=1
.Ltmp820:                               # EH_LABEL
	movq	%rax, %rbx
	movl	$15, %edx
	movq	%rax, %rdi
	leaq	.L.str.35(%rip), %rsi
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
.Ltmp821:                               # EH_LABEL
# %bb.327:                              # %_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc.exit629
                                        #   in Loop: Header=BB17_268 Depth=1
.Ltmp822:                               # EH_LABEL
	movq	%rbx, %rdi
	movsd	40(%rsp), %xmm0                 # 8-byte Reload
                                        # xmm0 = mem[0],zero
	callq	_ZNSo9_M_insertIdEERSoT_@PLT
.Ltmp823:                               # EH_LABEL
# %bb.328:                              # %_ZNSolsEd.exit631
                                        #   in Loop: Header=BB17_268 Depth=1
.Ltmp824:                               # EH_LABEL
	movq	%rax, %rbx
	movl	$7, %edx
	movq	%rax, %rdi
	leaq	.L.str.31(%rip), %rsi
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
.Ltmp825:                               # EH_LABEL
# %bb.329:                              # %_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc.exit633
                                        #   in Loop: Header=BB17_268 Depth=1
.Ltmp826:                               # EH_LABEL
	movq	%rbx, %rdi
	movsd	8(%rsp), %xmm0                  # 8-byte Reload
                                        # xmm0 = mem[0],zero
	callq	_ZNSo9_M_insertIdEERSoT_@PLT
.Ltmp827:                               # EH_LABEL
# %bb.330:                              # %_ZNSolsEd.exit635
                                        #   in Loop: Header=BB17_268 Depth=1
.Ltmp828:                               # EH_LABEL
	movq	%rax, %rbx
	movl	$1, %edx
	movq	%rax, %rdi
	leaq	.L.str.5(%rip), %rsi
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
.Ltmp829:                               # EH_LABEL
# %bb.331:                              # %_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc.exit637
                                        #   in Loop: Header=BB17_268 Depth=1
	movsd	8(%rsp), %xmm0                  # 8-byte Reload
                                        # xmm0 = mem[0],zero
	ucomisd	.LCPI17_0(%rip), %xmm0
	leaq	.L.str.25(%rip), %rsi
	leaq	.L.str.6(%rip), %rax
	cmovaeq	%rax, %rsi
.Ltmp830:                               # EH_LABEL
	movl	$4, %edx
	movq	%rbx, %rdi
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
.Ltmp831:                               # EH_LABEL
# %bb.332:                              # %_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc.exit640
                                        #   in Loop: Header=BB17_268 Depth=1
.Ltmp832:                               # EH_LABEL
	movl	$1, %edx
	movq	%rbx, %rdi
	leaq	.L.str.8(%rip), %rsi
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
.Ltmp833:                               # EH_LABEL
# %bb.333:                              # %_ZNSt6vectorIdSaIdEED2Ev.exit648
                                        #   in Loop: Header=BB17_268 Depth=1
	movq	16(%rsp), %rdi                  # 8-byte Reload
	movq	80(%rsp), %rsi                  # 8-byte Reload
	subq	%rdi, %rsi
	callq	_ZdlPvm@PLT
	movq	56(%rsp), %rdi                  # 8-byte Reload
	movq	48(%rsp), %rsi                  # 8-byte Reload
	subq	%rdi, %rsi
	callq	_ZdlPvm@PLT
	movq	24(%rsp), %rdi                  # 8-byte Reload
	movq	32(%rsp), %rsi                  # 8-byte Reload
	subq	%rdi, %rsi
	callq	_ZdlPvm@PLT
	movq	96(%rsp), %rcx                  # 8-byte Reload
	jmp	.LBB17_267
.LBB17_334:
.Ltmp835:                               # EH_LABEL
	movq	_ZSt4cout@GOTPCREL(%rip), %rdi
	leaq	.L.str.36(%rip), %rsi
	movl	$80, %edx
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
.Ltmp836:                               # EH_LABEL
# %bb.335:                              # %_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc.exit569
.Ltmp838:                               # EH_LABEL
	movl	$4096, %edi                     # imm = 0x1000
	movl	$4096, %esi                     # imm = 0x1000
	xorl	%edx, %edx
	movl	$300, %ecx                      # imm = 0x12C
	callq	_Z21bench_decode_isolatediibi
	movsd	%xmm0, (%rsp)                   # 8-byte Spill
.Ltmp839:                               # EH_LABEL
# %bb.336:
.Ltmp841:                               # EH_LABEL
	movl	$4096, %edi                     # imm = 0x1000
	movl	$4096, %esi                     # imm = 0x1000
	movl	$1, %edx
	movl	$300, %ecx                      # imm = 0x12C
	callq	_Z21bench_decode_isolatediibi
	movsd	%xmm0, 8(%rsp)                  # 8-byte Spill
.Ltmp842:                               # EH_LABEL
# %bb.337:
.Ltmp844:                               # EH_LABEL
	movq	_ZSt4cout@GOTPCREL(%rip), %rdi
	leaq	.L.str.37(%rip), %rsi
	movl	$17, %edx
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
.Ltmp845:                               # EH_LABEL
# %bb.338:                              # %_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc.exit656
.Ltmp846:                               # EH_LABEL
	movq	_ZSt4cout@GOTPCREL(%rip), %rdi
	movsd	(%rsp), %xmm0                   # 8-byte Reload
                                        # xmm0 = mem[0],zero
	callq	_ZNSo9_M_insertIdEERSoT_@PLT
.Ltmp847:                               # EH_LABEL
# %bb.339:                              # %_ZNSolsEd.exit658
.Ltmp848:                               # EH_LABEL
	movq	%rax, %rbx
	leaq	.L.str.38(%rip), %rsi
	movl	$14, %edx
	movq	%rax, %rdi
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
.Ltmp849:                               # EH_LABEL
# %bb.340:                              # %_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc.exit660
.Ltmp850:                               # EH_LABEL
	movsd	(%rsp), %xmm1                   # 8-byte Reload
                                        # xmm1 = mem[0],zero
	divsd	.LCPI17_1(%rip), %xmm1
	movsd	.LCPI17_4(%rip), %xmm0          # xmm0 = [1.6777216E+7,0.0E+0]
	divsd	%xmm1, %xmm0
	divsd	.LCPI17_3(%rip), %xmm0
	movq	%rbx, %rdi
	callq	_ZNSo9_M_insertIdEERSoT_@PLT
.Ltmp851:                               # EH_LABEL
# %bb.341:                              # %_ZNSolsEd.exit662
.Ltmp852:                               # EH_LABEL
	leaq	.L.str.39(%rip), %rsi
	movl	$6, %edx
	movq	%rax, %rdi
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
.Ltmp853:                               # EH_LABEL
# %bb.342:                              # %_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc.exit664
.Ltmp854:                               # EH_LABEL
	movq	_ZSt4cout@GOTPCREL(%rip), %rdi
	leaq	.L.str.40(%rip), %rsi
	movl	$26, %edx
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
.Ltmp855:                               # EH_LABEL
# %bb.343:                              # %_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc.exit666
.Ltmp856:                               # EH_LABEL
	movq	_ZSt4cout@GOTPCREL(%rip), %rdi
	movsd	8(%rsp), %xmm0                  # 8-byte Reload
                                        # xmm0 = mem[0],zero
	callq	_ZNSo9_M_insertIdEERSoT_@PLT
.Ltmp857:                               # EH_LABEL
# %bb.344:                              # %_ZNSolsEd.exit668
.Ltmp858:                               # EH_LABEL
	movq	%rax, %rbx
	leaq	.L.str.41(%rip), %rsi
	movl	$13, %edx
	movq	%rax, %rdi
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
.Ltmp859:                               # EH_LABEL
# %bb.345:                              # %_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc.exit670
.Ltmp860:                               # EH_LABEL
	movsd	8(%rsp), %xmm1                  # 8-byte Reload
                                        # xmm1 = mem[0],zero
	divsd	.LCPI17_1(%rip), %xmm1
	movsd	.LCPI17_5(%rip), %xmm0          # xmm0 = [8.388608E+6,0.0E+0]
	movsd	%xmm1, (%rsp)                   # 8-byte Spill
	divsd	%xmm1, %xmm0
	divsd	.LCPI17_3(%rip), %xmm0
	movq	%rbx, %rdi
	callq	_ZNSo9_M_insertIdEERSoT_@PLT
.Ltmp861:                               # EH_LABEL
# %bb.346:                              # %_ZNSolsEd.exit672
.Ltmp862:                               # EH_LABEL
	movq	%rax, %rbx
	leaq	.L.str.42(%rip), %rsi
	movl	$12, %edx
	movq	%rax, %rdi
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
.Ltmp863:                               # EH_LABEL
# %bb.347:                              # %_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc.exit674
.Ltmp864:                               # EH_LABEL
	movsd	.LCPI17_4(%rip), %xmm0          # xmm0 = [1.6777216E+7,0.0E+0]
	divsd	(%rsp), %xmm0                   # 8-byte Folded Reload
	divsd	.LCPI17_3(%rip), %xmm0
	movq	%rbx, %rdi
	callq	_ZNSo9_M_insertIdEERSoT_@PLT
.Ltmp865:                               # EH_LABEL
# %bb.348:                              # %_ZNSolsEd.exit676
.Ltmp866:                               # EH_LABEL
	leaq	.L.str.39(%rip), %rsi
	movl	$6, %edx
	movq	%rax, %rdi
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
.Ltmp867:                               # EH_LABEL
# %bb.349:                              # %_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc.exit678
.Ltmp869:                               # EH_LABEL
	movq	_ZSt4cout@GOTPCREL(%rip), %rdi
	leaq	.L.str.43(%rip), %rsi
	movl	$16, %edx
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
.Ltmp870:                               # EH_LABEL
# %bb.350:                              # %_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc.exit680
.Ltmp871:                               # EH_LABEL
	movq	_ZSt4cout@GOTPCREL(%rip), %rdi
	movsd	.LCPI17_6(%rip), %xmm0          # xmm0 = [5.17E+2,0.0E+0]
	callq	_ZNSo9_M_insertIdEERSoT_@PLT
.Ltmp872:                               # EH_LABEL
# %bb.351:                              # %_ZNSolsEd.exit682
.Ltmp873:                               # EH_LABEL
	movq	%rax, %rbx
	leaq	.L.str.44(%rip), %rsi
	movl	$8, %edx
	movq	%rax, %rdi
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
.Ltmp874:                               # EH_LABEL
# %bb.352:                              # %_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc.exit684
.Ltmp875:                               # EH_LABEL
	movsd	.LCPI17_7(%rip), %xmm0          # xmm0 = [1.6225547388781431E-2,0.0E+0]
	movq	%rbx, %rdi
	callq	_ZNSo9_M_insertIdEERSoT_@PLT
.Ltmp876:                               # EH_LABEL
# %bb.353:                              # %_ZNSolsEd.exit686
.Ltmp877:                               # EH_LABEL
	movq	%rax, %rbx
	leaq	.L.str.45(%rip), %rsi
	movl	$21, %edx
	movq	%rax, %rdi
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
.Ltmp878:                               # EH_LABEL
# %bb.354:                              # %_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc.exit688
.Ltmp879:                               # EH_LABEL
	movq	%rbx, %rdi
	movsd	8(%rsp), %xmm0                  # 8-byte Reload
                                        # xmm0 = mem[0],zero
	callq	_ZNSo9_M_insertIdEERSoT_@PLT
.Ltmp880:                               # EH_LABEL
# %bb.355:                              # %_ZNSolsEd.exit690
.Ltmp881:                               # EH_LABEL
	movq	%rax, %rbx
	leaq	.L.str.46(%rip), %rsi
	movl	$4, %edx
	movq	%rax, %rdi
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
.Ltmp882:                               # EH_LABEL
# %bb.356:                              # %_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc.exit692
	movsd	8(%rsp), %xmm0                  # 8-byte Reload
                                        # xmm0 = mem[0],zero
	ucomisd	.LCPI17_7(%rip), %xmm0
	leaq	.L.str.47(%rip), %rax
	leaq	.L.str.48(%rip), %rsi
	cmovaq	%rax, %rsi
	movl	$22, %eax
	movl	$8, %edx
	cmovaq	%rax, %rdx
.Ltmp883:                               # EH_LABEL
	movq	%rbx, %rdi
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
.Ltmp884:                               # EH_LABEL
# %bb.357:                              # %_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc.exit695
.Ltmp885:                               # EH_LABEL
	leaq	.L.str.8(%rip), %rsi
	movl	$1, %edx
	movq	%rbx, %rdi
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
.Ltmp886:                               # EH_LABEL
# %bb.358:                              # %_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc.exit697.preheader
	movl	$8388608, %ebx                  # imm = 0x800000
	.p2align	4
.LBB17_359:                             # %_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc.exit697
                                        # =>This Inner Loop Header: Depth=1
	callq	rand@PLT
	decq	%rbx
	jne	.LBB17_359
# %bb.360:
	callq	_ZNSt6chrono3_V212system_clock3nowEv@PLT
	movq	%rax, %rbx
	callq	_ZNSt6chrono3_V212system_clock3nowEv@PLT
	movq	%rax, %r14
.Ltmp888:                               # EH_LABEL
	movq	_ZSt4cout@GOTPCREL(%rip), %rdi
	leaq	.L.str.49(%rip), %rsi
	movl	$51, %edx
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
.Ltmp889:                               # EH_LABEL
# %bb.361:                              # %_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc.exit702
.Ltmp890:                               # EH_LABEL
	subq	%rbx, %r14
	xorps	%xmm0, %xmm0
	cvtsi2sd	%r14, %xmm0
	divsd	.LCPI17_8(%rip), %xmm0
	divsd	.LCPI17_9(%rip), %xmm0
	movq	_ZSt4cout@GOTPCREL(%rip), %rdi
	movsd	%xmm0, 8(%rsp)                  # 8-byte Spill
	callq	_ZNSo9_M_insertIdEERSoT_@PLT
.Ltmp891:                               # EH_LABEL
# %bb.362:                              # %_ZNSolsEd.exit704
.Ltmp892:                               # EH_LABEL
	movq	%rax, %rbx
	leaq	.L.str.50(%rip), %rsi
	movl	$10, %edx
	movq	%rax, %rdi
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
.Ltmp893:                               # EH_LABEL
# %bb.363:                              # %_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc.exit706
.Ltmp894:                               # EH_LABEL
	movq	%rbx, %rdi
	movl	$4096, %esi                     # imm = 0x1000
	callq	_ZNSolsEi@PLT
.Ltmp895:                               # EH_LABEL
# %bb.364:
.Ltmp896:                               # EH_LABEL
	movq	%rax, %rbx
	leaq	.L.str.51(%rip), %rsi
	movl	$3, %edx
	movq	%rax, %rdi
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
.Ltmp897:                               # EH_LABEL
# %bb.365:                              # %_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc.exit708
.Ltmp898:                               # EH_LABEL
	movq	%rbx, %rdi
	movl	$4096, %esi                     # imm = 0x1000
	callq	_ZNSolsEi@PLT
.Ltmp899:                               # EH_LABEL
# %bb.366:
.Ltmp900:                               # EH_LABEL
	movq	%rax, %rbx
	leaq	.L.str.52(%rip), %rsi
	movl	$2, %edx
	movq	%rax, %rdi
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
.Ltmp901:                               # EH_LABEL
# %bb.367:                              # %_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc.exit710
.Ltmp902:                               # EH_LABEL
	movq	%rbx, %rdi
	movl	$8, %esi
	callq	_ZNSolsEi@PLT
.Ltmp903:                               # EH_LABEL
# %bb.368:
.Ltmp904:                               # EH_LABEL
	movq	%rax, %rbx
	leaq	.L.str.53(%rip), %rsi
	movl	$8, %edx
	movq	%rax, %rdi
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
.Ltmp905:                               # EH_LABEL
# %bb.369:                              # %_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc.exit712
.Ltmp906:                               # EH_LABEL
	movsd	8(%rsp), %xmm1                  # 8-byte Reload
                                        # xmm1 = mem[0],zero
	divsd	.LCPI17_1(%rip), %xmm1
	movsd	.LCPI17_5(%rip), %xmm0          # xmm0 = [8.388608E+6,0.0E+0]
	divsd	%xmm1, %xmm0
	divsd	.LCPI17_3(%rip), %xmm0
	movq	%rbx, %rdi
	callq	_ZNSo9_M_insertIdEERSoT_@PLT
.Ltmp907:                               # EH_LABEL
# %bb.370:                              # %_ZNSolsEd.exit714
.Ltmp908:                               # EH_LABEL
	leaq	.L.str.39(%rip), %rsi
	movl	$6, %edx
	movq	%rax, %rdi
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
.Ltmp909:                               # EH_LABEL
# %bb.371:                              # %_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc.exit716
.Ltmp910:                               # EH_LABEL
	movq	_ZSt4cout@GOTPCREL(%rip), %rdi
	leaq	.L.str.54(%rip), %rsi
	movl	$119, %edx
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
.Ltmp911:                               # EH_LABEL
# %bb.372:                              # %_ZNSt6vectorIZ4mainE5ShapeSaIS0_EED2Ev.exit
	movl	$72, %esi
	movq	104(%rsp), %rdi                 # 8-byte Reload
	callq	_ZdlPvm@PLT
	xorl	%eax, %eax
	addq	$1592, %rsp                     # imm = 0x638
	.cfi_def_cfa_offset 56
	popq	%rbx
	.cfi_def_cfa_offset 48
	popq	%r12
	.cfi_def_cfa_offset 40
	popq	%r13
	.cfi_def_cfa_offset 32
	popq	%r14
	.cfi_def_cfa_offset 24
	popq	%r15
	.cfi_def_cfa_offset 16
	popq	%rbp
	.cfi_def_cfa_offset 8
	retq
.LBB17_373:
	.cfi_def_cfa_offset 1648
.Ltmp948:                               # EH_LABEL
	leaq	.L.str.57(%rip), %rdi
	movq	8(%rsp), %r13                   # 8-byte Reload
	callq	_ZSt20__throw_length_errorPKc@PLT
.Ltmp949:                               # EH_LABEL
# %bb.374:                              # %.noexc240
.LBB17_375:
.Ltmp945:                               # EH_LABEL
	leaq	.L.str.57(%rip), %rdi
	callq	_ZSt20__throw_length_errorPKc@PLT
.Ltmp946:                               # EH_LABEL
# %bb.376:                              # %.noexc249
.LBB17_377:
.Ltmp937:                               # EH_LABEL
	leaq	.L.str.57(%rip), %rdi
	callq	_ZSt20__throw_length_errorPKc@PLT
.Ltmp938:                               # EH_LABEL
# %bb.378:                              # %.noexc326
.LBB17_379:
.Ltmp942:                               # EH_LABEL
	leaq	.L.str.57(%rip), %rdi
	movq	32(%rsp), %r14                  # 8-byte Reload
	callq	_ZSt20__throw_length_errorPKc@PLT
.Ltmp943:                               # EH_LABEL
# %bb.380:                              # %.noexc307
.LBB17_381:
.Ltmp940:                               # EH_LABEL
	leaq	.L.str.57(%rip), %rdi
	movq	(%rsp), %rax                    # 8-byte Reload
	movq	%rax, 56(%rsp)                  # 8-byte Spill
	movq	16(%rsp), %rbp                  # 8-byte Reload
	callq	_ZSt20__throw_length_errorPKc@PLT
.Ltmp941:                               # EH_LABEL
# %bb.382:                              # %.noexc316
.LBB17_383:
.Ltmp929:                               # EH_LABEL
	leaq	.L.str.57(%rip), %rdi
	callq	_ZSt20__throw_length_errorPKc@PLT
.Ltmp930:                               # EH_LABEL
# %bb.384:                              # %.noexc413
.LBB17_385:
.Ltmp934:                               # EH_LABEL
	leaq	.L.str.57(%rip), %rdi
	movq	64(%rsp), %r14                  # 8-byte Reload
	callq	_ZSt20__throw_length_errorPKc@PLT
.Ltmp935:                               # EH_LABEL
# %bb.386:                              # %.noexc393
.LBB17_387:
.Ltmp932:                               # EH_LABEL
	leaq	.L.str.57(%rip), %rdi
	movq	64(%rsp), %r14                  # 8-byte Reload
	callq	_ZSt20__throw_length_errorPKc@PLT
.Ltmp933:                               # EH_LABEL
# %bb.388:                              # %.noexc403
.LBB17_389:
.Ltmp921:                               # EH_LABEL
	leaq	.L.str.57(%rip), %rdi
	callq	_ZSt20__throw_length_errorPKc@PLT
.Ltmp922:                               # EH_LABEL
# %bb.390:                              # %.noexc516
.LBB17_391:
.Ltmp926:                               # EH_LABEL
	leaq	.L.str.57(%rip), %rdi
	movq	(%rsp), %rbx                    # 8-byte Reload
	callq	_ZSt20__throw_length_errorPKc@PLT
.Ltmp927:                               # EH_LABEL
# %bb.392:                              # %.noexc496
.LBB17_393:
.Ltmp924:                               # EH_LABEL
	leaq	.L.str.57(%rip), %rdi
	callq	_ZSt20__throw_length_errorPKc@PLT
.Ltmp925:                               # EH_LABEL
# %bb.394:                              # %.noexc506
.LBB17_395:
.Ltmp913:                               # EH_LABEL
	movq	%rcx, 32(%rsp)                  # 8-byte Spill
	leaq	.L.str.57(%rip), %rdi
	callq	_ZSt20__throw_length_errorPKc@PLT
.Ltmp914:                               # EH_LABEL
# %bb.396:                              # %.noexc603
.LBB17_397:
.Ltmp918:                               # EH_LABEL
	leaq	.L.str.57(%rip), %rdi
	movq	24(%rsp), %rbx                  # 8-byte Reload
	callq	_ZSt20__throw_length_errorPKc@PLT
.Ltmp919:                               # EH_LABEL
# %bb.398:                              # %.noexc583
.LBB17_399:
.Ltmp916:                               # EH_LABEL
	leaq	.L.str.57(%rip), %rdi
	movq	24(%rsp), %rbx                  # 8-byte Reload
	callq	_ZSt20__throw_length_errorPKc@PLT
.Ltmp917:                               # EH_LABEL
# %bb.400:                              # %.noexc593
.LBB17_401:
.Ltmp843:                               # EH_LABEL
	jmp	.LBB17_407
.LBB17_402:
.Ltmp840:                               # EH_LABEL
	jmp	.LBB17_407
.LBB17_403:
.Ltmp837:                               # EH_LABEL
	jmp	.LBB17_407
.LBB17_404:
.Ltmp887:                               # EH_LABEL
	jmp	.LBB17_407
.LBB17_405:                             # %_ZNSt6vectorIhSaIhEED2Ev.exit724
.Ltmp912:                               # EH_LABEL
	jmp	.LBB17_407
.LBB17_406:
.Ltmp868:                               # EH_LABEL
.LBB17_407:                             # %_ZNSt6vectorIZ4mainE5ShapeSaIS0_EED2Ev.exit728
	movq	%rax, %rbp
	jmp	.LBB17_489
.LBB17_408:
.Ltmp794:                               # EH_LABEL
	jmp	.LBB17_437
.LBB17_409:
.Ltmp799:                               # EH_LABEL
	jmp	.LBB17_437
.LBB17_410:
.Ltmp734:                               # EH_LABEL
	jmp	.LBB17_462
.LBB17_411:
.Ltmp739:                               # EH_LABEL
	jmp	.LBB17_462
.LBB17_412:
.Ltmp667:                               # EH_LABEL
	movq	%rax, %rbp
	movq	%r13, %r14
	jmp	.LBB17_473
.LBB17_413:
.Ltmp662:                               # EH_LABEL
	jmp	.LBB17_471
.LBB17_414:
.Ltmp657:                               # EH_LABEL
	jmp	.LBB17_471
.LBB17_415:
.Ltmp597:                               # EH_LABEL
	jmp	.LBB17_453
.LBB17_416:
.Ltmp602:                               # EH_LABEL
	jmp	.LBB17_453
.LBB17_417:
.Ltmp540:                               # EH_LABEL
	movq	%rax, %rbp
	jmp	.LBB17_457
.LBB17_418:
.Ltmp568:                               # EH_LABEL
	movq	%rax, %rbp
	jmp	.LBB17_458
.LBB17_419:
.Ltmp577:                               # EH_LABEL
	movq	%rax, %rbp
	jmp	.LBB17_458
.LBB17_420:                             # %.loopexit1038
.Ltmp789:                               # EH_LABEL
	jmp	.LBB17_437
.LBB17_421:                             # %.loopexit.split-lp1039
.Ltmp915:                               # EH_LABEL
	jmp	.LBB17_437
.LBB17_422:                             # %.thread1002
.Ltmp834:                               # EH_LABEL
	movq	%rax, %rbp
	movq	24(%rsp), %rbx                  # 8-byte Reload
	jmp	.LBB17_439
.LBB17_423:                             # %.loopexit.split-lp
.Ltmp920:                               # EH_LABEL
	movq	%rax, %rbp
	jmp	.LBB17_438
.LBB17_424:
.Ltmp563:                               # EH_LABEL
	movq	%rax, %rbp
	jmp	.LBB17_457
.LBB17_425:
.Ltmp779:                               # EH_LABEL
	jmp	.LBB17_437
.LBB17_426:                             # %.loopexit1050
.Ltmp729:                               # EH_LABEL
	jmp	.LBB17_462
.LBB17_427:                             # %.loopexit.split-lp1051
.Ltmp923:                               # EH_LABEL
	jmp	.LBB17_446
.LBB17_428:                             # %.loopexit1062
.Ltmp652:                               # EH_LABEL
	jmp	.LBB17_471
.LBB17_429:                             # %.loopexit.split-lp1063
.Ltmp931:                               # EH_LABEL
	jmp	.LBB17_448
.LBB17_430:                             # %.loopexit1074
.Ltmp592:                               # EH_LABEL
	movq	%rax, %rbp
	movq	80(%rsp), %rax                  # 8-byte Reload
	movq	%rax, 48(%rsp)                  # 8-byte Spill
	jmp	.LBB17_479
.LBB17_431:                             # %.loopexit.split-lp1075
.Ltmp939:                               # EH_LABEL
	movq	%rax, %rbp
	movq	80(%rsp), %rax                  # 8-byte Reload
	movq	%rax, 48(%rsp)                  # 8-byte Spill
	jmp	.LBB17_480
.LBB17_432:                             # %.loopexit.split-lp1085
.Ltmp947:                               # EH_LABEL
	movq	%rax, %rbp
	movq	%rbx, 16(%rsp)                  # 8-byte Spill
	movq	80(%rsp), %r15                  # 8-byte Reload
	jmp	.LBB17_457
.LBB17_433:                             # %.loopexit.split-lp1080
.Ltmp950:                               # EH_LABEL
	movq	%rax, %rbp
	movq	%r12, 24(%rsp)                  # 8-byte Spill
	jmp	.LBB17_457
.LBB17_434:                             # %.thread987
.Ltmp774:                               # EH_LABEL
	movq	%rax, %rbp
	movq	(%rsp), %rbx                    # 8-byte Reload
	jmp	.LBB17_464
.LBB17_435:                             # %.thread956
.Ltmp637:                               # EH_LABEL
	movq	%rax, %rbp
	jmp	.LBB17_484
.LBB17_436:                             # %.loopexit
.Ltmp786:                               # EH_LABEL
.LBB17_437:
	movq	%rax, %rbp
	movq	24(%rsp), %rbx                  # 8-byte Reload
.LBB17_438:
	cmpq	$0, 16(%rsp)                    # 8-byte Folded Reload
	je	.LBB17_440
.LBB17_439:
	movq	16(%rsp), %rdi                  # 8-byte Reload
	movq	80(%rsp), %rsi                  # 8-byte Reload
	subq	%rdi, %rsi
	callq	_ZdlPvm@PLT
.LBB17_440:                             # %_ZNSt6vectorIdSaIdEED2Ev.exit650
	movq	56(%rsp), %rdi                  # 8-byte Reload
	testq	%rdi, %rdi
	je	.LBB17_442
# %bb.441:
	movq	48(%rsp), %rsi                  # 8-byte Reload
	subq	%rdi, %rsi
	callq	_ZdlPvm@PLT
.LBB17_442:                             # %_ZNSt6vectorIdSaIdEED2Ev.exit652
	testq	%rbx, %rbx
	je	.LBB17_489
# %bb.443:
	movq	32(%rsp), %rsi                  # 8-byte Reload
	jmp	.LBB17_469
.LBB17_444:
.Ltmp714:                               # EH_LABEL
	movq	%rax, %rbp
	movq	%r15, %r14
	jmp	.LBB17_473
.LBB17_445:                             # %.loopexit.split-lp1044
.Ltmp928:                               # EH_LABEL
.LBB17_446:
	movq	%rax, %rbp
	jmp	.LBB17_463
.LBB17_447:                             # %.loopexit.split-lp1056
.Ltmp936:                               # EH_LABEL
.LBB17_448:
	movq	%rax, %rbp
	jmp	.LBB17_472
.LBB17_449:                             # %.loopexit.split-lp1068
.Ltmp944:                               # EH_LABEL
	movq	%rbp, 16(%rsp)                  # 8-byte Spill
	movq	%rax, %rbp
	jmp	.LBB17_480
.LBB17_450:
.Ltmp719:                               # EH_LABEL
	jmp	.LBB17_462
.LBB17_451:
.Ltmp642:                               # EH_LABEL
	jmp	.LBB17_471
.LBB17_452:
.Ltmp582:                               # EH_LABEL
.LBB17_453:
	movq	%rax, %rbp
	jmp	.LBB17_481
.LBB17_454:                             # %.loopexit1084
.Ltmp535:                               # EH_LABEL
	movq	%rax, %rbp
	movq	80(%rsp), %r15                  # 8-byte Reload
	jmp	.LBB17_456
.LBB17_455:                             # %.loopexit1079
.Ltmp530:                               # EH_LABEL
	movq	%rax, %rbp
.LBB17_456:
	movq	8(%rsp), %r13                   # 8-byte Reload
.LBB17_457:
	testq	%r13, %r13
	je	.LBB17_459
.LBB17_458:                             # %.thread
	movq	16(%rsp), %rsi                  # 8-byte Reload
	subq	%r13, %rsi
	movq	%r13, %rdi
	callq	_ZdlPvm@PLT
.LBB17_459:                             # %_ZNSt6vectorIdSaIdEED2Ev.exit295
	testq	%r15, %r15
	je	.LBB17_489
# %bb.460:
	movq	24(%rsp), %rsi                  # 8-byte Reload
	subq	%r15, %rsi
	movq	%r15, %rdi
	jmp	.LBB17_488
.LBB17_461:                             # %.loopexit1043
.Ltmp726:                               # EH_LABEL
.LBB17_462:
	movq	%rax, %rbp
	movq	(%rsp), %rbx                    # 8-byte Reload
.LBB17_463:
	cmpq	$0, 72(%rsp)                    # 8-byte Folded Reload
	je	.LBB17_465
.LBB17_464:
	movq	72(%rsp), %rdi                  # 8-byte Reload
	movq	40(%rsp), %rsi                  # 8-byte Reload
	subq	%rdi, %rsi
	callq	_ZdlPvm@PLT
.LBB17_465:                             # %_ZNSt6vectorIdSaIdEED2Ev.exit563
	movq	64(%rsp), %rdi                  # 8-byte Reload
	testq	%rdi, %rdi
	movq	24(%rsp), %rsi                  # 8-byte Reload
	je	.LBB17_467
# %bb.466:
	subq	%rdi, %rsi
	callq	_ZdlPvm@PLT
.LBB17_467:                             # %_ZNSt6vectorIdSaIdEED2Ev.exit565
	testq	%rbx, %rbx
	je	.LBB17_489
# %bb.468:
	movq	16(%rsp), %rsi                  # 8-byte Reload
.LBB17_469:                             # %_ZNSt6vectorIZ4mainE5ShapeSaIS0_EED2Ev.exit728
	subq	%rbx, %rsi
	movq	%rbx, %rdi
	jmp	.LBB17_488
.LBB17_470:                             # %.loopexit1055
.Ltmp649:                               # EH_LABEL
.LBB17_471:
	movq	%rax, %rbp
	movq	64(%rsp), %r14                  # 8-byte Reload
.LBB17_472:
	cmpq	$0, 56(%rsp)                    # 8-byte Folded Reload
	je	.LBB17_474
.LBB17_473:                             # %.thread972
	movq	56(%rsp), %rdi                  # 8-byte Reload
	movq	80(%rsp), %rsi                  # 8-byte Reload
	subq	%rdi, %rsi
	callq	_ZdlPvm@PLT
.LBB17_474:                             # %_ZNSt6vectorIdSaIdEED2Ev.exit476
	movq	72(%rsp), %rdi                  # 8-byte Reload
	testq	%rdi, %rdi
	je	.LBB17_476
# %bb.475:
	movq	48(%rsp), %rsi                  # 8-byte Reload
	subq	%rdi, %rsi
	callq	_ZdlPvm@PLT
.LBB17_476:                             # %_ZNSt6vectorIdSaIdEED2Ev.exit478
	testq	%r14, %r14
	je	.LBB17_489
# %bb.477:
	movq	24(%rsp), %rsi                  # 8-byte Reload
	jmp	.LBB17_487
.LBB17_478:                             # %.loopexit1067
.Ltmp589:                               # EH_LABEL
	movq	%rax, %rbp
.LBB17_479:
	movq	32(%rsp), %r14                  # 8-byte Reload
.LBB17_480:
	movq	%r12, %r13
.LBB17_481:
	cmpq	$0, 88(%rsp)                    # 8-byte Folded Reload
	jne	.LBB17_484
# %bb.482:                              # %_ZNSt6vectorIdSaIdEED2Ev.exit373
	testq	%r13, %r13
	jne	.LBB17_485
.LBB17_483:                             # %_ZNSt6vectorIdSaIdEED2Ev.exit375
	testq	%r14, %r14
	jne	.LBB17_486
	jmp	.LBB17_489
.LBB17_484:
	movq	88(%rsp), %rdi                  # 8-byte Reload
	movq	56(%rsp), %rsi                  # 8-byte Reload
	subq	%rdi, %rsi
	callq	_ZdlPvm@PLT
	testq	%r13, %r13
	je	.LBB17_483
.LBB17_485:
	movq	16(%rsp), %rsi                  # 8-byte Reload
	subq	%r13, %rsi
	movq	%r13, %rdi
	callq	_ZdlPvm@PLT
	testq	%r14, %r14
	je	.LBB17_489
.LBB17_486:
	movq	48(%rsp), %rsi                  # 8-byte Reload
.LBB17_487:                             # %_ZNSt6vectorIZ4mainE5ShapeSaIS0_EED2Ev.exit728
	subq	%r14, %rsi
	movq	%r14, %rdi
.LBB17_488:                             # %_ZNSt6vectorIZ4mainE5ShapeSaIS0_EED2Ev.exit728
	callq	_ZdlPvm@PLT
.LBB17_489:                             # %_ZNSt6vectorIZ4mainE5ShapeSaIS0_EED2Ev.exit728
	movl	$72, %esi
	movq	104(%rsp), %rdi                 # 8-byte Reload
	callq	_ZdlPvm@PLT
	movq	%rbp, %rdi
	callq	_Unwind_Resume@PLT
.Lfunc_end17:
	.size	main, .Lfunc_end17-main
	.cfi_endproc
	.section	.gcc_except_table,"a",@progbits
	.p2align	2, 0x0
GCC_except_table17:
.Lexception5:
	.byte	255                             # @LPStart Encoding = omit
	.byte	255                             # @TType Encoding = omit
	.byte	1                               # Call site Encoding = uleb128
	.uleb128 .Lcst_end5-.Lcst_begin5
.Lcst_begin5:
	.uleb128 .Lfunc_begin5-.Lfunc_begin5    # >> Call Site 1 <<
	.uleb128 .Ltmp524-.Lfunc_begin5         #   Call between .Lfunc_begin5 and .Ltmp524
	.byte	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp524-.Lfunc_begin5         # >> Call Site 2 <<
	.uleb128 .Ltmp525-.Ltmp524              #   Call between .Ltmp524 and .Ltmp525
	.uleb128 .Ltmp837-.Lfunc_begin5         #     jumps to .Ltmp837
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp526-.Lfunc_begin5         # >> Call Site 3 <<
	.uleb128 .Ltmp529-.Ltmp526              #   Call between .Ltmp526 and .Ltmp529
	.uleb128 .Ltmp530-.Lfunc_begin5         #     jumps to .Ltmp530
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp529-.Lfunc_begin5         # >> Call Site 4 <<
	.uleb128 .Ltmp531-.Ltmp529              #   Call between .Ltmp529 and .Ltmp531
	.byte	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp531-.Lfunc_begin5         # >> Call Site 5 <<
	.uleb128 .Ltmp534-.Ltmp531              #   Call between .Ltmp531 and .Ltmp534
	.uleb128 .Ltmp535-.Lfunc_begin5         #     jumps to .Ltmp535
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp534-.Lfunc_begin5         # >> Call Site 6 <<
	.uleb128 .Ltmp536-.Ltmp534              #   Call between .Ltmp534 and .Ltmp536
	.byte	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp536-.Lfunc_begin5         # >> Call Site 7 <<
	.uleb128 .Ltmp539-.Ltmp536              #   Call between .Ltmp536 and .Ltmp539
	.uleb128 .Ltmp540-.Lfunc_begin5         #     jumps to .Ltmp540
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp541-.Lfunc_begin5         # >> Call Site 8 <<
	.uleb128 .Ltmp562-.Ltmp541              #   Call between .Ltmp541 and .Ltmp562
	.uleb128 .Ltmp563-.Lfunc_begin5         #     jumps to .Ltmp563
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp564-.Lfunc_begin5         # >> Call Site 9 <<
	.uleb128 .Ltmp567-.Ltmp564              #   Call between .Ltmp564 and .Ltmp567
	.uleb128 .Ltmp568-.Lfunc_begin5         #     jumps to .Ltmp568
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp569-.Lfunc_begin5         # >> Call Site 10 <<
	.uleb128 .Ltmp576-.Ltmp569              #   Call between .Ltmp569 and .Ltmp576
	.uleb128 .Ltmp577-.Lfunc_begin5         #     jumps to .Ltmp577
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp578-.Lfunc_begin5         # >> Call Site 11 <<
	.uleb128 .Ltmp579-.Ltmp578              #   Call between .Ltmp578 and .Ltmp579
	.uleb128 .Ltmp837-.Lfunc_begin5         #     jumps to .Ltmp837
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp580-.Lfunc_begin5         # >> Call Site 12 <<
	.uleb128 .Ltmp581-.Ltmp580              #   Call between .Ltmp580 and .Ltmp581
	.uleb128 .Ltmp582-.Lfunc_begin5         #     jumps to .Ltmp582
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp583-.Lfunc_begin5         # >> Call Site 13 <<
	.uleb128 .Ltmp586-.Ltmp583              #   Call between .Ltmp583 and .Ltmp586
	.uleb128 .Ltmp589-.Lfunc_begin5         #     jumps to .Ltmp589
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp586-.Lfunc_begin5         # >> Call Site 14 <<
	.uleb128 .Ltmp587-.Ltmp586              #   Call between .Ltmp586 and .Ltmp587
	.byte	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp587-.Lfunc_begin5         # >> Call Site 15 <<
	.uleb128 .Ltmp588-.Ltmp587              #   Call between .Ltmp587 and .Ltmp588
	.uleb128 .Ltmp589-.Lfunc_begin5         #     jumps to .Ltmp589
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp588-.Lfunc_begin5         # >> Call Site 16 <<
	.uleb128 .Ltmp590-.Ltmp588              #   Call between .Ltmp588 and .Ltmp590
	.byte	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp590-.Lfunc_begin5         # >> Call Site 17 <<
	.uleb128 .Ltmp591-.Ltmp590              #   Call between .Ltmp590 and .Ltmp591
	.uleb128 .Ltmp592-.Lfunc_begin5         #     jumps to .Ltmp592
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp591-.Lfunc_begin5         # >> Call Site 18 <<
	.uleb128 .Ltmp593-.Ltmp591              #   Call between .Ltmp591 and .Ltmp593
	.byte	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp593-.Lfunc_begin5         # >> Call Site 19 <<
	.uleb128 .Ltmp596-.Ltmp593              #   Call between .Ltmp593 and .Ltmp596
	.uleb128 .Ltmp597-.Lfunc_begin5         #     jumps to .Ltmp597
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp598-.Lfunc_begin5         # >> Call Site 20 <<
	.uleb128 .Ltmp601-.Ltmp598              #   Call between .Ltmp598 and .Ltmp601
	.uleb128 .Ltmp602-.Lfunc_begin5         #     jumps to .Ltmp602
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp603-.Lfunc_begin5         # >> Call Site 21 <<
	.uleb128 .Ltmp636-.Ltmp603              #   Call between .Ltmp603 and .Ltmp636
	.uleb128 .Ltmp637-.Lfunc_begin5         #     jumps to .Ltmp637
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp638-.Lfunc_begin5         # >> Call Site 22 <<
	.uleb128 .Ltmp639-.Ltmp638              #   Call between .Ltmp638 and .Ltmp639
	.uleb128 .Ltmp837-.Lfunc_begin5         #     jumps to .Ltmp837
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp640-.Lfunc_begin5         # >> Call Site 23 <<
	.uleb128 .Ltmp641-.Ltmp640              #   Call between .Ltmp640 and .Ltmp641
	.uleb128 .Ltmp642-.Lfunc_begin5         #     jumps to .Ltmp642
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp643-.Lfunc_begin5         # >> Call Site 24 <<
	.uleb128 .Ltmp646-.Ltmp643              #   Call between .Ltmp643 and .Ltmp646
	.uleb128 .Ltmp649-.Lfunc_begin5         #     jumps to .Ltmp649
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp646-.Lfunc_begin5         # >> Call Site 25 <<
	.uleb128 .Ltmp647-.Ltmp646              #   Call between .Ltmp646 and .Ltmp647
	.byte	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp647-.Lfunc_begin5         # >> Call Site 26 <<
	.uleb128 .Ltmp648-.Ltmp647              #   Call between .Ltmp647 and .Ltmp648
	.uleb128 .Ltmp649-.Lfunc_begin5         #     jumps to .Ltmp649
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp648-.Lfunc_begin5         # >> Call Site 27 <<
	.uleb128 .Ltmp650-.Ltmp648              #   Call between .Ltmp648 and .Ltmp650
	.byte	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp650-.Lfunc_begin5         # >> Call Site 28 <<
	.uleb128 .Ltmp651-.Ltmp650              #   Call between .Ltmp650 and .Ltmp651
	.uleb128 .Ltmp652-.Lfunc_begin5         #     jumps to .Ltmp652
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp651-.Lfunc_begin5         # >> Call Site 29 <<
	.uleb128 .Ltmp653-.Ltmp651              #   Call between .Ltmp651 and .Ltmp653
	.byte	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp653-.Lfunc_begin5         # >> Call Site 30 <<
	.uleb128 .Ltmp656-.Ltmp653              #   Call between .Ltmp653 and .Ltmp656
	.uleb128 .Ltmp657-.Lfunc_begin5         #     jumps to .Ltmp657
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp658-.Lfunc_begin5         # >> Call Site 31 <<
	.uleb128 .Ltmp661-.Ltmp658              #   Call between .Ltmp658 and .Ltmp661
	.uleb128 .Ltmp662-.Lfunc_begin5         #     jumps to .Ltmp662
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp663-.Lfunc_begin5         # >> Call Site 32 <<
	.uleb128 .Ltmp666-.Ltmp663              #   Call between .Ltmp663 and .Ltmp666
	.uleb128 .Ltmp667-.Lfunc_begin5         #     jumps to .Ltmp667
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp668-.Lfunc_begin5         # >> Call Site 33 <<
	.uleb128 .Ltmp713-.Ltmp668              #   Call between .Ltmp668 and .Ltmp713
	.uleb128 .Ltmp714-.Lfunc_begin5         #     jumps to .Ltmp714
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp715-.Lfunc_begin5         # >> Call Site 34 <<
	.uleb128 .Ltmp716-.Ltmp715              #   Call between .Ltmp715 and .Ltmp716
	.uleb128 .Ltmp837-.Lfunc_begin5         #     jumps to .Ltmp837
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp717-.Lfunc_begin5         # >> Call Site 35 <<
	.uleb128 .Ltmp718-.Ltmp717              #   Call between .Ltmp717 and .Ltmp718
	.uleb128 .Ltmp719-.Lfunc_begin5         #     jumps to .Ltmp719
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp720-.Lfunc_begin5         # >> Call Site 36 <<
	.uleb128 .Ltmp723-.Ltmp720              #   Call between .Ltmp720 and .Ltmp723
	.uleb128 .Ltmp726-.Lfunc_begin5         #     jumps to .Ltmp726
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp723-.Lfunc_begin5         # >> Call Site 37 <<
	.uleb128 .Ltmp724-.Ltmp723              #   Call between .Ltmp723 and .Ltmp724
	.byte	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp724-.Lfunc_begin5         # >> Call Site 38 <<
	.uleb128 .Ltmp725-.Ltmp724              #   Call between .Ltmp724 and .Ltmp725
	.uleb128 .Ltmp726-.Lfunc_begin5         #     jumps to .Ltmp726
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp725-.Lfunc_begin5         # >> Call Site 39 <<
	.uleb128 .Ltmp727-.Ltmp725              #   Call between .Ltmp725 and .Ltmp727
	.byte	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp727-.Lfunc_begin5         # >> Call Site 40 <<
	.uleb128 .Ltmp728-.Ltmp727              #   Call between .Ltmp727 and .Ltmp728
	.uleb128 .Ltmp729-.Lfunc_begin5         #     jumps to .Ltmp729
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp728-.Lfunc_begin5         # >> Call Site 41 <<
	.uleb128 .Ltmp730-.Ltmp728              #   Call between .Ltmp728 and .Ltmp730
	.byte	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp730-.Lfunc_begin5         # >> Call Site 42 <<
	.uleb128 .Ltmp733-.Ltmp730              #   Call between .Ltmp730 and .Ltmp733
	.uleb128 .Ltmp734-.Lfunc_begin5         #     jumps to .Ltmp734
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp735-.Lfunc_begin5         # >> Call Site 43 <<
	.uleb128 .Ltmp738-.Ltmp735              #   Call between .Ltmp735 and .Ltmp738
	.uleb128 .Ltmp739-.Lfunc_begin5         #     jumps to .Ltmp739
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp740-.Lfunc_begin5         # >> Call Site 44 <<
	.uleb128 .Ltmp773-.Ltmp740              #   Call between .Ltmp740 and .Ltmp773
	.uleb128 .Ltmp774-.Lfunc_begin5         #     jumps to .Ltmp774
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp775-.Lfunc_begin5         # >> Call Site 45 <<
	.uleb128 .Ltmp776-.Ltmp775              #   Call between .Ltmp775 and .Ltmp776
	.uleb128 .Ltmp837-.Lfunc_begin5         #     jumps to .Ltmp837
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp777-.Lfunc_begin5         # >> Call Site 46 <<
	.uleb128 .Ltmp778-.Ltmp777              #   Call between .Ltmp777 and .Ltmp778
	.uleb128 .Ltmp779-.Lfunc_begin5         #     jumps to .Ltmp779
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp780-.Lfunc_begin5         # >> Call Site 47 <<
	.uleb128 .Ltmp783-.Ltmp780              #   Call between .Ltmp780 and .Ltmp783
	.uleb128 .Ltmp786-.Lfunc_begin5         #     jumps to .Ltmp786
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp783-.Lfunc_begin5         # >> Call Site 48 <<
	.uleb128 .Ltmp784-.Ltmp783              #   Call between .Ltmp783 and .Ltmp784
	.byte	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp784-.Lfunc_begin5         # >> Call Site 49 <<
	.uleb128 .Ltmp785-.Ltmp784              #   Call between .Ltmp784 and .Ltmp785
	.uleb128 .Ltmp786-.Lfunc_begin5         #     jumps to .Ltmp786
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp785-.Lfunc_begin5         # >> Call Site 50 <<
	.uleb128 .Ltmp787-.Ltmp785              #   Call between .Ltmp785 and .Ltmp787
	.byte	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp787-.Lfunc_begin5         # >> Call Site 51 <<
	.uleb128 .Ltmp788-.Ltmp787              #   Call between .Ltmp787 and .Ltmp788
	.uleb128 .Ltmp789-.Lfunc_begin5         #     jumps to .Ltmp789
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp788-.Lfunc_begin5         # >> Call Site 52 <<
	.uleb128 .Ltmp790-.Ltmp788              #   Call between .Ltmp788 and .Ltmp790
	.byte	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp790-.Lfunc_begin5         # >> Call Site 53 <<
	.uleb128 .Ltmp793-.Ltmp790              #   Call between .Ltmp790 and .Ltmp793
	.uleb128 .Ltmp794-.Lfunc_begin5         #     jumps to .Ltmp794
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp795-.Lfunc_begin5         # >> Call Site 54 <<
	.uleb128 .Ltmp798-.Ltmp795              #   Call between .Ltmp795 and .Ltmp798
	.uleb128 .Ltmp799-.Lfunc_begin5         #     jumps to .Ltmp799
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp800-.Lfunc_begin5         # >> Call Site 55 <<
	.uleb128 .Ltmp833-.Ltmp800              #   Call between .Ltmp800 and .Ltmp833
	.uleb128 .Ltmp834-.Lfunc_begin5         #     jumps to .Ltmp834
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp835-.Lfunc_begin5         # >> Call Site 56 <<
	.uleb128 .Ltmp836-.Ltmp835              #   Call between .Ltmp835 and .Ltmp836
	.uleb128 .Ltmp837-.Lfunc_begin5         #     jumps to .Ltmp837
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp838-.Lfunc_begin5         # >> Call Site 57 <<
	.uleb128 .Ltmp839-.Ltmp838              #   Call between .Ltmp838 and .Ltmp839
	.uleb128 .Ltmp840-.Lfunc_begin5         #     jumps to .Ltmp840
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp841-.Lfunc_begin5         # >> Call Site 58 <<
	.uleb128 .Ltmp842-.Ltmp841              #   Call between .Ltmp841 and .Ltmp842
	.uleb128 .Ltmp843-.Lfunc_begin5         #     jumps to .Ltmp843
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp844-.Lfunc_begin5         # >> Call Site 59 <<
	.uleb128 .Ltmp867-.Ltmp844              #   Call between .Ltmp844 and .Ltmp867
	.uleb128 .Ltmp868-.Lfunc_begin5         #     jumps to .Ltmp868
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp869-.Lfunc_begin5         # >> Call Site 60 <<
	.uleb128 .Ltmp886-.Ltmp869              #   Call between .Ltmp869 and .Ltmp886
	.uleb128 .Ltmp887-.Lfunc_begin5         #     jumps to .Ltmp887
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp888-.Lfunc_begin5         # >> Call Site 61 <<
	.uleb128 .Ltmp911-.Ltmp888              #   Call between .Ltmp888 and .Ltmp911
	.uleb128 .Ltmp912-.Lfunc_begin5         #     jumps to .Ltmp912
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp948-.Lfunc_begin5         # >> Call Site 62 <<
	.uleb128 .Ltmp949-.Ltmp948              #   Call between .Ltmp948 and .Ltmp949
	.uleb128 .Ltmp950-.Lfunc_begin5         #     jumps to .Ltmp950
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp945-.Lfunc_begin5         # >> Call Site 63 <<
	.uleb128 .Ltmp946-.Ltmp945              #   Call between .Ltmp945 and .Ltmp946
	.uleb128 .Ltmp947-.Lfunc_begin5         #     jumps to .Ltmp947
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp937-.Lfunc_begin5         # >> Call Site 64 <<
	.uleb128 .Ltmp938-.Ltmp937              #   Call between .Ltmp937 and .Ltmp938
	.uleb128 .Ltmp939-.Lfunc_begin5         #     jumps to .Ltmp939
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp942-.Lfunc_begin5         # >> Call Site 65 <<
	.uleb128 .Ltmp941-.Ltmp942              #   Call between .Ltmp942 and .Ltmp941
	.uleb128 .Ltmp944-.Lfunc_begin5         #     jumps to .Ltmp944
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp929-.Lfunc_begin5         # >> Call Site 66 <<
	.uleb128 .Ltmp930-.Ltmp929              #   Call between .Ltmp929 and .Ltmp930
	.uleb128 .Ltmp931-.Lfunc_begin5         #     jumps to .Ltmp931
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp934-.Lfunc_begin5         # >> Call Site 67 <<
	.uleb128 .Ltmp933-.Ltmp934              #   Call between .Ltmp934 and .Ltmp933
	.uleb128 .Ltmp936-.Lfunc_begin5         #     jumps to .Ltmp936
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp921-.Lfunc_begin5         # >> Call Site 68 <<
	.uleb128 .Ltmp922-.Ltmp921              #   Call between .Ltmp921 and .Ltmp922
	.uleb128 .Ltmp923-.Lfunc_begin5         #     jumps to .Ltmp923
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp926-.Lfunc_begin5         # >> Call Site 69 <<
	.uleb128 .Ltmp925-.Ltmp926              #   Call between .Ltmp926 and .Ltmp925
	.uleb128 .Ltmp928-.Lfunc_begin5         #     jumps to .Ltmp928
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp913-.Lfunc_begin5         # >> Call Site 70 <<
	.uleb128 .Ltmp914-.Ltmp913              #   Call between .Ltmp913 and .Ltmp914
	.uleb128 .Ltmp915-.Lfunc_begin5         #     jumps to .Ltmp915
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp918-.Lfunc_begin5         # >> Call Site 71 <<
	.uleb128 .Ltmp917-.Ltmp918              #   Call between .Ltmp918 and .Ltmp917
	.uleb128 .Ltmp920-.Lfunc_begin5         #     jumps to .Ltmp920
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp917-.Lfunc_begin5         # >> Call Site 72 <<
	.uleb128 .Lfunc_end17-.Ltmp917          #   Call between .Ltmp917 and .Lfunc_end17
	.byte	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
.Lcst_end5:
	.p2align	2, 0x0
                                        # -- End function
	.section	.text._ZSt16__introsort_loopIN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEElNS0_5__ops15_Iter_less_iterEEvT_S9_T0_T1_,"axG",@progbits,_ZSt16__introsort_loopIN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEElNS0_5__ops15_Iter_less_iterEEvT_S9_T0_T1_,comdat
	.weak	_ZSt16__introsort_loopIN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEElNS0_5__ops15_Iter_less_iterEEvT_S9_T0_T1_ # -- Begin function _ZSt16__introsort_loopIN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEElNS0_5__ops15_Iter_less_iterEEvT_S9_T0_T1_
	.prefalign	4, .Lfunc_end18, nop
	.type	_ZSt16__introsort_loopIN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEElNS0_5__ops15_Iter_less_iterEEvT_S9_T0_T1_,@function
_ZSt16__introsort_loopIN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEElNS0_5__ops15_Iter_less_iterEEvT_S9_T0_T1_: # @_ZSt16__introsort_loopIN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEElNS0_5__ops15_Iter_less_iterEEvT_S9_T0_T1_
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	pushq	%r15
	.cfi_def_cfa_offset 24
	pushq	%r14
	.cfi_def_cfa_offset 32
	pushq	%r13
	.cfi_def_cfa_offset 40
	pushq	%r12
	.cfi_def_cfa_offset 48
	pushq	%rbx
	.cfi_def_cfa_offset 56
	pushq	%rax
	.cfi_def_cfa_offset 64
	.cfi_offset %rbx, -56
	.cfi_offset %r12, -48
	.cfi_offset %r13, -40
	.cfi_offset %r14, -32
	.cfi_offset %r15, -24
	.cfi_offset %rbp, -16
	movq	%rsi, %r12
	subq	%rdi, %r12
	movq	%r12, %rax
	sarq	$3, %rax
	cmpq	$17, %rax
	jl	.LBB18_41
# %bb.1:                                # %.lr.ph
	movq	%rdx, %r14
	movq	%rdi, %rbx
	testq	%rdx, %rdx
	je	.LBB18_24
# %bb.2:                                # %.lr.ph63.preheader
	movq	$-8, %rbp
	subq	%rbx, %rbp
	.p2align	4
.LBB18_3:                               # %.lr.ph63
                                        # =>This Loop Header: Depth=1
                                        #     Child Loop BB18_15 Depth 2
                                        #       Child Loop BB18_16 Depth 3
                                        #       Child Loop BB18_18 Depth 3
	shrq	%rax
	movsd	8(%rbx), %xmm1                  # xmm1 = mem[0],zero
	movsd	(%rbx,%rax,8), %xmm2            # xmm2 = mem[0],zero
	ucomisd	%xmm1, %xmm2
	movsd	-8(%rsi), %xmm0                 # xmm0 = mem[0],zero
	jbe	.LBB18_6
# %bb.4:                                #   in Loop: Header=BB18_3 Depth=1
	ucomisd	%xmm2, %xmm0
	jbe	.LBB18_8
# %bb.5:                                #   in Loop: Header=BB18_3 Depth=1
	movsd	(%rbx), %xmm0                   # xmm0 = mem[0],zero
	movsd	%xmm2, (%rbx)
	movsd	%xmm0, (%rbx,%rax,8)
	jmp	.LBB18_14
	.p2align	4
.LBB18_6:                               #   in Loop: Header=BB18_3 Depth=1
	ucomisd	%xmm1, %xmm0
	jbe	.LBB18_10
# %bb.7:                                #   in Loop: Header=BB18_3 Depth=1
	movsd	(%rbx), %xmm0                   # xmm0 = mem[0],zero
	movsd	%xmm1, (%rbx)
	movsd	%xmm0, 8(%rbx)
	jmp	.LBB18_14
	.p2align	4
.LBB18_8:                               #   in Loop: Header=BB18_3 Depth=1
	ucomisd	%xmm1, %xmm0
	movsd	(%rbx), %xmm2                   # xmm2 = mem[0],zero
	jbe	.LBB18_12
# %bb.9:                                #   in Loop: Header=BB18_3 Depth=1
	movsd	%xmm0, (%rbx)
	movsd	%xmm2, -8(%rsi)
	jmp	.LBB18_14
	.p2align	4
.LBB18_10:                              #   in Loop: Header=BB18_3 Depth=1
	ucomisd	%xmm2, %xmm0
	movsd	(%rbx), %xmm1                   # xmm1 = mem[0],zero
	jbe	.LBB18_13
# %bb.11:                               #   in Loop: Header=BB18_3 Depth=1
	movsd	%xmm0, (%rbx)
	movsd	%xmm1, -8(%rsi)
	jmp	.LBB18_14
.LBB18_12:                              #   in Loop: Header=BB18_3 Depth=1
	movsd	%xmm1, (%rbx)
	movsd	%xmm2, 8(%rbx)
	jmp	.LBB18_14
.LBB18_13:                              #   in Loop: Header=BB18_3 Depth=1
	movsd	%xmm2, (%rbx)
	movsd	%xmm1, (%rbx,%rax,8)
	.p2align	4
.LBB18_14:                              # %_ZSt22__move_median_to_firstIN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEENS0_5__ops15_Iter_less_iterEEvT_S9_S9_S9_T0_.exit.i.preheader
                                        #   in Loop: Header=BB18_3 Depth=1
	decq	%r14
	leaq	8(%rbx), %r13
	movq	%rsi, %rax
	.p2align	4
.LBB18_15:                              # %_ZSt22__move_median_to_firstIN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEENS0_5__ops15_Iter_less_iterEEvT_S9_S9_S9_T0_.exit.i
                                        #   Parent Loop BB18_3 Depth=1
                                        # =>  This Loop Header: Depth=2
                                        #       Child Loop BB18_16 Depth 3
                                        #       Child Loop BB18_18 Depth 3
	movsd	(%rbx), %xmm0                   # xmm0 = mem[0],zero
	movq	%rbp, %r12
	addq	%r13, %r12
	.p2align	4
.LBB18_16:                              #   Parent Loop BB18_3 Depth=1
                                        #     Parent Loop BB18_15 Depth=2
                                        # =>    This Inner Loop Header: Depth=3
	movsd	(%r13), %xmm1                   # xmm1 = mem[0],zero
	addq	$8, %r13
	addq	$8, %r12
	ucomisd	%xmm1, %xmm0
	ja	.LBB18_16
# %bb.17:                               # %.preheader.i.i.preheader
                                        #   in Loop: Header=BB18_15 Depth=2
	leaq	-8(%r13), %r15
	.p2align	4
.LBB18_18:                              # %.preheader.i.i
                                        #   Parent Loop BB18_3 Depth=1
                                        #     Parent Loop BB18_15 Depth=2
                                        # =>    This Inner Loop Header: Depth=3
	movsd	-8(%rax), %xmm2                 # xmm2 = mem[0],zero
	addq	$-8, %rax
	ucomisd	%xmm0, %xmm2
	ja	.LBB18_18
# %bb.19:                               #   in Loop: Header=BB18_15 Depth=2
	cmpq	%rax, %r15
	jae	.LBB18_21
# %bb.20:                               #   in Loop: Header=BB18_15 Depth=2
	movsd	%xmm2, (%r15)
	movsd	%xmm1, (%rax)
	jmp	.LBB18_15
	.p2align	4
.LBB18_21:                              # %_ZSt27__unguarded_partition_pivotIN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEENS0_5__ops15_Iter_less_iterEET_S9_S9_T0_.exit
                                        #   in Loop: Header=BB18_3 Depth=1
	movq	%r15, %rdi
	movq	%r14, %rdx
	callq	_ZSt16__introsort_loopIN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEElNS0_5__ops15_Iter_less_iterEEvT_S9_T0_T1_
	movq	%r12, %rax
	sarq	$3, %rax
	cmpq	$16, %rax
	jle	.LBB18_41
# %bb.22:                               #   in Loop: Header=BB18_3 Depth=1
	movq	%r15, %rsi
	testq	%r14, %r14
	jne	.LBB18_3
# %bb.23:                               # %._crit_edge.loopexit
	addq	$-8, %r13
	movq	%r13, %rsi
.LBB18_24:                              # %._crit_edge
	leaq	-2(%rax), %rcx
	shrq	%rcx
	decq	%rax
	movq	%rax, %rdx
	shrq	%rdx
	movq	%rcx, %rdi
	jmp	.LBB18_27
	.p2align	4
.LBB18_25:                              #   in Loop: Header=BB18_27 Depth=1
	movq	%r8, %r9
.LBB18_26:                              # %_ZSt13__adjust_heapIN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEEldNS0_5__ops15_Iter_less_iterEEvT_T0_SA_T1_T2_.exit.i.i
                                        #   in Loop: Header=BB18_27 Depth=1
	movsd	%xmm0, (%rbx,%r9,8)
	subq	$1, %rdi
	jb	.LBB18_40
.LBB18_27:                              # =>This Loop Header: Depth=1
                                        #     Child Loop BB18_31 Depth 2
                                        #     Child Loop BB18_37 Depth 2
	movsd	(%rbx,%rdi,8), %xmm0            # xmm0 = mem[0],zero
	movq	%rdi, %r8
	cmpq	%rdx, %rdi
	jge	.LBB18_33
# %bb.28:                               # %.lr.ph.i.i.i.preheader
                                        #   in Loop: Header=BB18_27 Depth=1
	movq	%rdi, %r9
	jmp	.LBB18_31
	.p2align	4
.LBB18_29:                              # %.lr.ph.i.i.i
                                        #   in Loop: Header=BB18_31 Depth=2
	leaq	2(,%r9,2), %r8
.LBB18_30:                              # %.lr.ph.i.i.i
                                        #   in Loop: Header=BB18_31 Depth=2
	movsd	(%rbx,%r8,8), %xmm1             # xmm1 = mem[0],zero
	movsd	%xmm1, (%rbx,%r9,8)
	movq	%r8, %r9
	cmpq	%rdx, %r8
	jge	.LBB18_33
.LBB18_31:                              # %.lr.ph.i.i.i
                                        #   Parent Loop BB18_27 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	leaq	(%r9,%r9), %r8
	movsd	8(%rbx,%r8,8), %xmm1            # xmm1 = mem[0],zero
	ucomisd	16(%rbx,%r8,8), %xmm1
	jbe	.LBB18_29
# %bb.32:                               #   in Loop: Header=BB18_31 Depth=2
	leaq	1(,%r9,2), %r8
	jmp	.LBB18_30
	.p2align	4
.LBB18_33:                              # %._crit_edge.i.i.i
                                        #   in Loop: Header=BB18_27 Depth=1
	testb	$8, %r12b
	jne	.LBB18_36
# %bb.34:                               # %._crit_edge.i.i.i
                                        #   in Loop: Header=BB18_27 Depth=1
	cmpq	%rcx, %r8
	jne	.LBB18_36
# %bb.35:                               #   in Loop: Header=BB18_27 Depth=1
	movsd	(%rbx,%rax,8), %xmm1            # xmm1 = mem[0],zero
	movsd	%xmm1, (%rbx,%rcx,8)
	movq	%rax, %r8
.LBB18_36:                              #   in Loop: Header=BB18_27 Depth=1
	cmpq	%rdi, %r8
	jle	.LBB18_25
	.p2align	4
.LBB18_37:                              # %.lr.ph.i.i.i.i11
                                        #   Parent Loop BB18_27 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	leaq	-1(%r8), %r9
	shrq	$63, %r9
	addq	%r8, %r9
	decq	%r9
	sarq	%r9
	movsd	(%rbx,%r9,8), %xmm1             # xmm1 = mem[0],zero
	ucomisd	%xmm1, %xmm0
	jbe	.LBB18_25
# %bb.38:                               #   in Loop: Header=BB18_37 Depth=2
	movsd	%xmm1, (%rbx,%r8,8)
	movq	%r9, %r8
	cmpq	%rdi, %r9
	jg	.LBB18_37
	jmp	.LBB18_26
.LBB18_40:                              # %_ZSt13__heap_selectIN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEENS0_5__ops15_Iter_less_iterEEvT_S9_S9_T0_.exit
	cmpq	$9, %r12
	jge	.LBB18_44
.LBB18_41:                              # %_ZSt14__partial_sortIN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEENS0_5__ops15_Iter_less_iterEEvT_S9_S9_T0_.exit
	addq	$8, %rsp
	.cfi_def_cfa_offset 56
	popq	%rbx
	.cfi_def_cfa_offset 48
	popq	%r12
	.cfi_def_cfa_offset 40
	popq	%r13
	.cfi_def_cfa_offset 32
	popq	%r14
	.cfi_def_cfa_offset 24
	popq	%r15
	.cfi_def_cfa_offset 16
	popq	%rbp
	.cfi_def_cfa_offset 8
	retq
	.p2align	4
.LBB18_42:                              #   in Loop: Header=BB18_44 Depth=1
	.cfi_def_cfa_offset 64
	xorl	%ecx, %ecx
.LBB18_43:                              # %_ZSt10__pop_heapIN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEENS0_5__ops15_Iter_less_iterEEvT_S9_S9_RT0_.exit.i.i
                                        #   in Loop: Header=BB18_44 Depth=1
	movsd	%xmm0, (%rbx,%rcx,8)
	cmpq	$8, %rax
	jle	.LBB18_41
.LBB18_44:                              # %.lr.ph.i.i
                                        # =>This Loop Header: Depth=1
                                        #     Child Loop BB18_48 Depth 2
                                        #     Child Loop BB18_55 Depth 2
	movsd	-8(%rsi), %xmm0                 # xmm0 = mem[0],zero
	movsd	(%rbx), %xmm1                   # xmm1 = mem[0],zero
	movsd	%xmm1, -8(%rsi)
	addq	$-8, %rsi
	movq	%rsi, %rax
	subq	%rbx, %rax
	movq	%rax, %rdx
	sarq	$3, %rdx
	cmpq	$3, %rdx
	jl	.LBB18_50
# %bb.45:                               # %.lr.ph.i.i.i.i.preheader
                                        #   in Loop: Header=BB18_44 Depth=1
	leaq	-1(%rdx), %rcx
	shrq	$63, %rcx
	leaq	(%rdx,%rcx), %rdi
	decq	%rdi
	sarq	%rdi
	xorl	%r8d, %r8d
	jmp	.LBB18_48
	.p2align	4
.LBB18_46:                              # %.lr.ph.i.i.i.i
                                        #   in Loop: Header=BB18_48 Depth=2
	leaq	2(,%r8,2), %rcx
.LBB18_47:                              # %.lr.ph.i.i.i.i
                                        #   in Loop: Header=BB18_48 Depth=2
	movsd	(%rbx,%rcx,8), %xmm1            # xmm1 = mem[0],zero
	movsd	%xmm1, (%rbx,%r8,8)
	movq	%rcx, %r8
	cmpq	%rdi, %rcx
	jge	.LBB18_51
.LBB18_48:                              # %.lr.ph.i.i.i.i
                                        #   Parent Loop BB18_44 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	leaq	(%r8,%r8), %rcx
	movsd	8(%rbx,%rcx,8), %xmm1           # xmm1 = mem[0],zero
	ucomisd	16(%rbx,%rcx,8), %xmm1
	jbe	.LBB18_46
# %bb.49:                               #   in Loop: Header=BB18_48 Depth=2
	leaq	1(,%r8,2), %rcx
	jmp	.LBB18_47
	.p2align	4
.LBB18_50:                              #   in Loop: Header=BB18_44 Depth=1
	xorl	%ecx, %ecx
.LBB18_51:                              # %._crit_edge.i.i.i.i
                                        #   in Loop: Header=BB18_44 Depth=1
	testb	$8, %al
	jne	.LBB18_54
# %bb.52:                               #   in Loop: Header=BB18_44 Depth=1
	addq	$-2, %rdx
	sarq	%rdx
	cmpq	%rdx, %rcx
	jne	.LBB18_54
# %bb.53:                               # %.thread.i.i.i
                                        #   in Loop: Header=BB18_44 Depth=1
	leaq	(%rcx,%rcx), %rdx
	movsd	8(%rbx,%rdx,8), %xmm1           # xmm1 = mem[0],zero
	movsd	%xmm1, (%rbx,%rcx,8)
	leaq	1(,%rcx,2), %rcx
	jmp	.LBB18_55
	.p2align	4
.LBB18_54:                              #   in Loop: Header=BB18_44 Depth=1
	testq	%rcx, %rcx
	je	.LBB18_42
	.p2align	4
.LBB18_55:                              # %.lr.ph.i.i.i.i.i
                                        #   Parent Loop BB18_44 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	leaq	-1(%rcx), %rdx
	shrq	%rdx
	movsd	(%rbx,%rdx,8), %xmm1            # xmm1 = mem[0],zero
	ucomisd	%xmm1, %xmm0
	jbe	.LBB18_43
# %bb.56:                               #   in Loop: Header=BB18_55 Depth=2
	movsd	%xmm1, (%rbx,%rcx,8)
	movq	%rdx, %rcx
	testq	%rdx, %rdx
	jne	.LBB18_55
	jmp	.LBB18_42
.Lfunc_end18:
	.size	_ZSt16__introsort_loopIN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEElNS0_5__ops15_Iter_less_iterEEvT_S9_T0_T1_, .Lfunc_end18-_ZSt16__introsort_loopIN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEElNS0_5__ops15_Iter_less_iterEEvT_S9_T0_T1_
	.cfi_endproc
                                        # -- End function
	.section	.text._ZSt22__final_insertion_sortIN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEENS0_5__ops15_Iter_less_iterEEvT_S9_T0_,"axG",@progbits,_ZSt22__final_insertion_sortIN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEENS0_5__ops15_Iter_less_iterEEvT_S9_T0_,comdat
	.weak	_ZSt22__final_insertion_sortIN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEENS0_5__ops15_Iter_less_iterEEvT_S9_T0_ # -- Begin function _ZSt22__final_insertion_sortIN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEENS0_5__ops15_Iter_less_iterEEvT_S9_T0_
	.prefalign	4, .Lfunc_end19, nop
	.type	_ZSt22__final_insertion_sortIN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEENS0_5__ops15_Iter_less_iterEEvT_S9_T0_,@function
_ZSt22__final_insertion_sortIN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEENS0_5__ops15_Iter_less_iterEEvT_S9_T0_: # @_ZSt22__final_insertion_sortIN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEENS0_5__ops15_Iter_less_iterEEvT_S9_T0_
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	pushq	%r15
	.cfi_def_cfa_offset 24
	pushq	%r14
	.cfi_def_cfa_offset 32
	pushq	%r13
	.cfi_def_cfa_offset 40
	pushq	%r12
	.cfi_def_cfa_offset 48
	pushq	%rbx
	.cfi_def_cfa_offset 56
	pushq	%rax
	.cfi_def_cfa_offset 64
	.cfi_offset %rbx, -56
	.cfi_offset %r12, -48
	.cfi_offset %r13, -40
	.cfi_offset %r14, -32
	.cfi_offset %r15, -24
	.cfi_offset %rbp, -16
	movq	%rsi, %rbx
	movq	%rdi, %r14
	movq	%rsi, %rax
	subq	%rdi, %rax
	cmpq	$129, %rax
	jl	.LBB19_17
# %bb.1:                                # %.lr.ph.i
	leaq	8(%r14), %r15
	movl	$8, %r12d
	movq	%r15, %r13
	movq	%r14, %rbp
	jmp	.LBB19_2
.LBB19_17:
	cmpq	%rbx, %r14
	je	.LBB19_29
# %bb.18:
	leaq	8(%r14), %rax
	cmpq	%rbx, %rax
	je	.LBB19_29
# %bb.19:                               # %.lr.ph.i15.preheader
	movq	%r14, %r15
	jmp	.LBB19_20
	.p2align	4
.LBB19_28:                              # %_ZSt13move_backwardIN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEES6_ET0_T_S8_S7_.exit.i18
                                        #   in Loop: Header=BB19_20 Depth=1
	movsd	%xmm1, (%rax)
	leaq	8(%r15), %rax
	cmpq	%rbx, %rax
	je	.LBB19_29
.LBB19_20:                              # %.lr.ph.i15
                                        # =>This Loop Header: Depth=1
                                        #     Child Loop BB19_27 Depth 2
	movq	%r15, %rdi
	movq	%rax, %r15
	movsd	8(%rdi), %xmm1                  # xmm1 = mem[0],zero
	movsd	(%r14), %xmm0                   # xmm0 = mem[0],zero
	ucomisd	%xmm1, %xmm0
	jbe	.LBB19_25
# %bb.21:                               # %_ZSt7advanceIPdlEvRT_T0_.exit.i.i.i.i.i28
                                        #   in Loop: Header=BB19_20 Depth=1
	movsd	%xmm1, (%rsp)                   # 8-byte Spill
	movq	%r15, %rdx
	subq	%r14, %rdx
	subq	%rdx, %rdi
	movq	%rdx, %rax
	sarq	$3, %rax
	addq	$16, %rdi
	cmpq	$2, %rax
	jl	.LBB19_23
# %bb.22:                               #   in Loop: Header=BB19_20 Depth=1
	movq	%r14, %rsi
	callq	memmove@PLT
	movq	%r14, %rax
	movsd	(%rsp), %xmm1                   # 8-byte Reload
                                        # xmm1 = mem[0],zero
	jmp	.LBB19_28
	.p2align	4
.LBB19_25:                              #   in Loop: Header=BB19_20 Depth=1
	movsd	(%rdi), %xmm0                   # xmm0 = mem[0],zero
	ucomisd	%xmm1, %xmm0
	movq	%r15, %rax
	jbe	.LBB19_28
# %bb.26:                               # %.lr.ph.i.i22.preheader
                                        #   in Loop: Header=BB19_20 Depth=1
	movq	%r15, %rax
	.p2align	4
.LBB19_27:                              # %.lr.ph.i.i22
                                        #   Parent Loop BB19_20 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	movsd	%xmm0, (%rax)
	movsd	-16(%rax), %xmm0                # xmm0 = mem[0],zero
	addq	$-8, %rax
	ucomisd	%xmm1, %xmm0
	ja	.LBB19_27
	jmp	.LBB19_28
.LBB19_23:                              # %_ZSt7advanceIPdlEvRT_T0_.exit.thread.i.i.i.i.i29
                                        #   in Loop: Header=BB19_20 Depth=1
	movq	%r14, %rax
	cmpq	$8, %rdx
	movsd	(%rsp), %xmm1                   # 8-byte Reload
                                        # xmm1 = mem[0],zero
	jne	.LBB19_28
# %bb.24:                               #   in Loop: Header=BB19_20 Depth=1
	movsd	%xmm0, (%rdi)
	movq	%r14, %rax
	jmp	.LBB19_28
.LBB19_5:                               # %_ZSt7advanceIPdlEvRT_T0_.exit.thread.i.i.i.i.i
                                        #   in Loop: Header=BB19_2 Depth=1
	movsd	%xmm0, (%r15)
	.p2align	4
.LBB19_6:                               # %_ZSt13move_backwardIN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEES6_ET0_T_S8_S7_.exit.i
                                        #   in Loop: Header=BB19_2 Depth=1
	movq	%r14, %rax
	movsd	(%rsp), %xmm1                   # 8-byte Reload
                                        # xmm1 = mem[0],zero
.LBB19_10:                              # %_ZSt13move_backwardIN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEES6_ET0_T_S8_S7_.exit.i
                                        #   in Loop: Header=BB19_2 Depth=1
	movsd	%xmm1, (%rax)
	addq	$8, %r12
	addq	$8, %r13
	cmpq	$128, %r12
	je	.LBB19_11
.LBB19_2:                               # =>This Loop Header: Depth=1
                                        #     Child Loop BB19_9 Depth 2
	movq	%rbp, %rax
	leaq	(%r14,%r12), %rbp
	movsd	(%r14,%r12), %xmm1              # xmm1 = mem[0],zero
	movsd	(%r14), %xmm0                   # xmm0 = mem[0],zero
	ucomisd	%xmm1, %xmm0
	jbe	.LBB19_7
# %bb.3:                                # %_ZSt7advanceIPdlEvRT_T0_.exit.i.i.i.i.i
                                        #   in Loop: Header=BB19_2 Depth=1
	movsd	%xmm1, (%rsp)                   # 8-byte Spill
	cmpq	$9, %r12
	jb	.LBB19_5
# %bb.4:                                #   in Loop: Header=BB19_2 Depth=1
	movq	%r15, %rdi
	movq	%r14, %rsi
	movq	%r12, %rdx
	callq	memmove@PLT
	jmp	.LBB19_6
	.p2align	4
.LBB19_7:                               #   in Loop: Header=BB19_2 Depth=1
	movsd	(%rax), %xmm0                   # xmm0 = mem[0],zero
	ucomisd	%xmm1, %xmm0
	movq	%rbp, %rax
	jbe	.LBB19_10
# %bb.8:                                # %.lr.ph.i.i.preheader
                                        #   in Loop: Header=BB19_2 Depth=1
	movq	%r13, %rax
	.p2align	4
.LBB19_9:                               # %.lr.ph.i.i
                                        #   Parent Loop BB19_2 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	movsd	%xmm0, (%rax)
	movsd	-16(%rax), %xmm0                # xmm0 = mem[0],zero
	addq	$-8, %rax
	ucomisd	%xmm1, %xmm0
	ja	.LBB19_9
	jmp	.LBB19_10
.LBB19_11:                              # %_ZSt16__insertion_sortIN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEENS0_5__ops15_Iter_less_iterEEvT_S9_T0_.exit
	subq	$-128, %r14
	jmp	.LBB19_12
	.p2align	4
.LBB19_16:                              # %_ZSt25__unguarded_linear_insertIN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEENS0_5__ops14_Val_less_iterEEvT_T0_.exit.i
                                        #   in Loop: Header=BB19_12 Depth=1
	movsd	%xmm0, (%rax)
	addq	$8, %r14
.LBB19_12:                              # %_ZSt16__insertion_sortIN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEENS0_5__ops15_Iter_less_iterEEvT_S9_T0_.exit
                                        # =>This Loop Header: Depth=1
                                        #     Child Loop BB19_15 Depth 2
	cmpq	%rbx, %r14
	je	.LBB19_29
# %bb.13:                               # %.lr.ph.i6
                                        #   in Loop: Header=BB19_12 Depth=1
	movsd	-8(%r14), %xmm1                 # xmm1 = mem[0],zero
	movsd	(%r14), %xmm0                   # xmm0 = mem[0],zero
	ucomisd	%xmm0, %xmm1
	movq	%r14, %rax
	jbe	.LBB19_16
# %bb.14:                               # %.lr.ph.i.i8.preheader
                                        #   in Loop: Header=BB19_12 Depth=1
	movq	%r14, %rax
	.p2align	4
.LBB19_15:                              # %.lr.ph.i.i8
                                        #   Parent Loop BB19_12 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	movsd	%xmm1, (%rax)
	movsd	-16(%rax), %xmm1                # xmm1 = mem[0],zero
	addq	$-8, %rax
	ucomisd	%xmm0, %xmm1
	ja	.LBB19_15
	jmp	.LBB19_16
.LBB19_29:                              # %_ZSt26__unguarded_insertion_sortIN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEENS0_5__ops15_Iter_less_iterEEvT_S9_T0_.exit
	addq	$8, %rsp
	.cfi_def_cfa_offset 56
	popq	%rbx
	.cfi_def_cfa_offset 48
	popq	%r12
	.cfi_def_cfa_offset 40
	popq	%r13
	.cfi_def_cfa_offset 32
	popq	%r14
	.cfi_def_cfa_offset 24
	popq	%r15
	.cfi_def_cfa_offset 16
	popq	%rbp
	.cfi_def_cfa_offset 8
	retq
.Lfunc_end19:
	.size	_ZSt22__final_insertion_sortIN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEENS0_5__ops15_Iter_less_iterEEvT_S9_T0_, .Lfunc_end19-_ZSt22__final_insertion_sortIN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEENS0_5__ops15_Iter_less_iterEEvT_S9_T0_
	.cfi_endproc
                                        # -- End function
	.text
	.prefalign	4, .Lfunc_end20, nop    # -- Begin function __hip_module_ctor
	.type	__hip_module_ctor,@function
__hip_module_ctor:                      # @__hip_module_ctor
	.cfi_startproc
# %bb.0:
	pushq	%rbx
	.cfi_def_cfa_offset 16
	subq	$32, %rsp
	.cfi_def_cfa_offset 48
	.cfi_offset %rbx, -16
	movq	__hip_gpubin_handle_437800c2e6994a55(%rip), %rbx
	testq	%rbx, %rbx
	jne	.LBB20_2
# %bb.1:
	leaq	__hip_fatbin_wrapper(%rip), %rdi
	callq	__hipRegisterFatBinary@PLT
	movq	%rax, %rbx
	movq	%rax, __hip_gpubin_handle_437800c2e6994a55(%rip)
.LBB20_2:
	xorps	%xmm0, %xmm0
	movups	%xmm0, 16(%rsp)
	movups	%xmm0, (%rsp)
	movq	_Z14plain_direct16PK14__hip_fp8_e4m3S1_Pfiii@GOTPCREL(%rip), %rsi
	leaq	.L__unnamed_1(%rip), %rcx
	movq	%rbx, %rdi
	movq	%rcx, %rdx
	movl	$-1, %r8d
	xorl	%r9d, %r9d
	callq	__hipRegisterFunction@PLT
	xorps	%xmm0, %xmm0
	movups	%xmm0, 16(%rsp)
	movups	%xmm0, (%rsp)
	movq	_Z14plain_direct32PK14__hip_fp8_e4m3S1_Pfiii@GOTPCREL(%rip), %rsi
	leaq	.L__unnamed_2(%rip), %rcx
	movq	%rbx, %rdi
	movq	%rcx, %rdx
	movl	$-1, %r8d
	xorl	%r9d, %r9d
	callq	__hipRegisterFunction@PLT
	xorps	%xmm0, %xmm0
	movups	%xmm0, 16(%rsp)
	movups	%xmm0, (%rsp)
	movq	_Z12fused_m5_k32PK14__hip_fp8_e4m3PKhS3_Pfiiii@GOTPCREL(%rip), %rsi
	leaq	.L__unnamed_3(%rip), %rcx
	movq	%rbx, %rdi
	movq	%rcx, %rdx
	movl	$-1, %r8d
	xorl	%r9d, %r9d
	callq	__hipRegisterFunction@PLT
	xorps	%xmm0, %xmm0
	movups	%xmm0, 16(%rsp)
	movups	%xmm0, (%rsp)
	movq	_Z21fused_m6_repacked_k32PK14__hip_fp8_e4m3PKhS3_Pfiii@GOTPCREL(%rip), %rsi
	leaq	.L__unnamed_4(%rip), %rcx
	movq	%rbx, %rdi
	movq	%rcx, %rdx
	movl	$-1, %r8d
	xorl	%r9d, %r9d
	callq	__hipRegisterFunction@PLT
	xorps	%xmm0, %xmm0
	movups	%xmm0, 16(%rsp)
	movups	%xmm0, (%rsp)
	movq	_Z24fused_m6_repacked_db_k32PK14__hip_fp8_e4m3PKhS3_Pfiii@GOTPCREL(%rip), %rsi
	leaq	.L__unnamed_5(%rip), %rcx
	movq	%rbx, %rdi
	movq	%rcx, %rdx
	movl	$-1, %r8d
	xorl	%r9d, %r9d
	callq	__hipRegisterFunction@PLT
	xorps	%xmm0, %xmm0
	movups	%xmm0, 16(%rsp)
	movups	%xmm0, (%rsp)
	movq	_Z21fused_m6_repacked_k64PK14__hip_fp8_e4m3PKhS3_Pfiii@GOTPCREL(%rip), %rsi
	leaq	.L__unnamed_6(%rip), %rcx
	movq	%rbx, %rdi
	movq	%rcx, %rdx
	movl	$-1, %r8d
	xorl	%r9d, %r9d
	callq	__hipRegisterFunction@PLT
	xorps	%xmm0, %xmm0
	movups	%xmm0, 16(%rsp)
	movups	%xmm0, (%rsp)
	movq	_Z21decode_only_scatteredPKhS0_Phiii@GOTPCREL(%rip), %rsi
	leaq	.L__unnamed_7(%rip), %rcx
	movq	%rbx, %rdi
	movq	%rcx, %rdx
	movl	$-1, %r8d
	xorl	%r9d, %r9d
	callq	__hipRegisterFunction@PLT
	xorps	%xmm0, %xmm0
	movups	%xmm0, 16(%rsp)
	movups	%xmm0, (%rsp)
	movq	_Z20decode_only_repackedPKhS0_Phii@GOTPCREL(%rip), %rsi
	leaq	.L__unnamed_8(%rip), %rcx
	movq	%rbx, %rdi
	movq	%rcx, %rdx
	movl	$-1, %r8d
	xorl	%r9d, %r9d
	callq	__hipRegisterFunction@PLT
	leaq	__hip_module_dtor(%rip), %rdi
	addq	$32, %rsp
	.cfi_def_cfa_offset 16
	popq	%rbx
	.cfi_def_cfa_offset 8
	jmp	atexit@PLT                      # TAILCALL
.Lfunc_end20:
	.size	__hip_module_ctor, .Lfunc_end20-__hip_module_ctor
	.cfi_endproc
                                        # -- End function
	.prefalign	4, .Lfunc_end21, nop    # -- Begin function __hip_module_dtor
	.type	__hip_module_dtor,@function
__hip_module_dtor:                      # @__hip_module_dtor
	.cfi_startproc
# %bb.0:
	movq	__hip_gpubin_handle_437800c2e6994a55(%rip), %rdi
	testq	%rdi, %rdi
	je	.LBB21_2
# %bb.1:
	pushq	%rax
	.cfi_def_cfa_offset 16
	callq	__hipUnregisterFatBinary@PLT
	movq	$0, __hip_gpubin_handle_437800c2e6994a55(%rip)
	addq	$8, %rsp
	.cfi_def_cfa_offset 8
.LBB21_2:
	retq
.Lfunc_end21:
	.size	__hip_module_dtor, .Lfunc_end21-__hip_module_dtor
	.cfi_endproc
                                        # -- End function
	.type	_Z14plain_direct16PK14__hip_fp8_e4m3S1_Pfiii,@object # @_Z14plain_direct16PK14__hip_fp8_e4m3S1_Pfiii
	.section	.data.rel.ro,"aw",@progbits
	.globl	_Z14plain_direct16PK14__hip_fp8_e4m3S1_Pfiii
	.p2align	3, 0x0
_Z14plain_direct16PK14__hip_fp8_e4m3S1_Pfiii:
	.quad	_Z29__device_stub__plain_direct16PK14__hip_fp8_e4m3S1_Pfiii
	.size	_Z14plain_direct16PK14__hip_fp8_e4m3S1_Pfiii, 8

	.type	_Z14plain_direct32PK14__hip_fp8_e4m3S1_Pfiii,@object # @_Z14plain_direct32PK14__hip_fp8_e4m3S1_Pfiii
	.globl	_Z14plain_direct32PK14__hip_fp8_e4m3S1_Pfiii
	.p2align	3, 0x0
_Z14plain_direct32PK14__hip_fp8_e4m3S1_Pfiii:
	.quad	_Z29__device_stub__plain_direct32PK14__hip_fp8_e4m3S1_Pfiii
	.size	_Z14plain_direct32PK14__hip_fp8_e4m3S1_Pfiii, 8

	.type	_Z12fused_m5_k32PK14__hip_fp8_e4m3PKhS3_Pfiiii,@object # @_Z12fused_m5_k32PK14__hip_fp8_e4m3PKhS3_Pfiiii
	.globl	_Z12fused_m5_k32PK14__hip_fp8_e4m3PKhS3_Pfiiii
	.p2align	3, 0x0
_Z12fused_m5_k32PK14__hip_fp8_e4m3PKhS3_Pfiiii:
	.quad	_Z27__device_stub__fused_m5_k32PK14__hip_fp8_e4m3PKhS3_Pfiiii
	.size	_Z12fused_m5_k32PK14__hip_fp8_e4m3PKhS3_Pfiiii, 8

	.type	_Z21fused_m6_repacked_k32PK14__hip_fp8_e4m3PKhS3_Pfiii,@object # @_Z21fused_m6_repacked_k32PK14__hip_fp8_e4m3PKhS3_Pfiii
	.globl	_Z21fused_m6_repacked_k32PK14__hip_fp8_e4m3PKhS3_Pfiii
	.p2align	3, 0x0
_Z21fused_m6_repacked_k32PK14__hip_fp8_e4m3PKhS3_Pfiii:
	.quad	_Z36__device_stub__fused_m6_repacked_k32PK14__hip_fp8_e4m3PKhS3_Pfiii
	.size	_Z21fused_m6_repacked_k32PK14__hip_fp8_e4m3PKhS3_Pfiii, 8

	.type	_Z24fused_m6_repacked_db_k32PK14__hip_fp8_e4m3PKhS3_Pfiii,@object # @_Z24fused_m6_repacked_db_k32PK14__hip_fp8_e4m3PKhS3_Pfiii
	.globl	_Z24fused_m6_repacked_db_k32PK14__hip_fp8_e4m3PKhS3_Pfiii
	.p2align	3, 0x0
_Z24fused_m6_repacked_db_k32PK14__hip_fp8_e4m3PKhS3_Pfiii:
	.quad	_Z39__device_stub__fused_m6_repacked_db_k32PK14__hip_fp8_e4m3PKhS3_Pfiii
	.size	_Z24fused_m6_repacked_db_k32PK14__hip_fp8_e4m3PKhS3_Pfiii, 8

	.type	_Z21fused_m6_repacked_k64PK14__hip_fp8_e4m3PKhS3_Pfiii,@object # @_Z21fused_m6_repacked_k64PK14__hip_fp8_e4m3PKhS3_Pfiii
	.globl	_Z21fused_m6_repacked_k64PK14__hip_fp8_e4m3PKhS3_Pfiii
	.p2align	3, 0x0
_Z21fused_m6_repacked_k64PK14__hip_fp8_e4m3PKhS3_Pfiii:
	.quad	_Z36__device_stub__fused_m6_repacked_k64PK14__hip_fp8_e4m3PKhS3_Pfiii
	.size	_Z21fused_m6_repacked_k64PK14__hip_fp8_e4m3PKhS3_Pfiii, 8

	.type	_Z21decode_only_scatteredPKhS0_Phiii,@object # @_Z21decode_only_scatteredPKhS0_Phiii
	.globl	_Z21decode_only_scatteredPKhS0_Phiii
	.p2align	3, 0x0
_Z21decode_only_scatteredPKhS0_Phiii:
	.quad	_Z36__device_stub__decode_only_scatteredPKhS0_Phiii
	.size	_Z21decode_only_scatteredPKhS0_Phiii, 8

	.type	_Z20decode_only_repackedPKhS0_Phii,@object # @_Z20decode_only_repackedPKhS0_Phii
	.globl	_Z20decode_only_repackedPKhS0_Phii
	.p2align	3, 0x0
_Z20decode_only_repackedPKhS0_Phii:
	.quad	_Z35__device_stub__decode_only_repackedPKhS0_Phii
	.size	_Z20decode_only_repackedPKhS0_Phii, 8

	.type	.L.str,@object                  # @.str
	.section	.rodata.str1.1,"aMS",@progbits,1
.L.str:
	.asciz	"M6 validate repacked_k32 M"
	.size	.L.str, 27

	.type	.L.str.1,@object                # @.str.1
.L.str.1:
	.asciz	" N"
	.size	.L.str.1, 3

	.type	.L.str.2,@object                # @.str.2
.L.str.2:
	.asciz	" K"
	.size	.L.str.2, 3

	.type	.L.str.3,@object                # @.str.3
.L.str.3:
	.asciz	" max_abs "
	.size	.L.str.3, 10

	.type	.L.str.4,@object                # @.str.4
.L.str.4:
	.asciz	" mism "
	.size	.L.str.4, 7

	.type	.L.str.5,@object                # @.str.5
.L.str.5:
	.asciz	" "
	.size	.L.str.5, 2

	.type	.L.str.6,@object                # @.str.6
.L.str.6:
	.asciz	"PASS"
	.size	.L.str.6, 5

	.type	.L.str.7,@object                # @.str.7
.L.str.7:
	.asciz	"FAIL"
	.size	.L.str.7, 5

	.type	.L.str.8,@object                # @.str.8
.L.str.8:
	.asciz	"\n"
	.size	.L.str.8, 2

	.type	.L.str.9,@object                # @.str.9
.L.str.9:
	.asciz	"M6 validate repacked_db_k32 max_abs "
	.size	.L.str.9, 37

	.type	.L.str.10,@object               # @.str.10
.L.str.10:
	.asciz	"M6 validate repacked_k64 max_abs "
	.size	.L.str.10, 34

	.type	.L.str.11,@object               # @.str.11
.L.str.11:
	.asciz	"Device "
	.size	.L.str.11, 8

	.type	.L.str.12,@object               # @.str.12
.L.str.12:
	.asciz	" CUs\n"
	.size	.L.str.12, 6

	.type	.L.str.13,@object               # @.str.13
.L.str.13:
	.asciz	"VALIDATION FAILED - bench still runs for diagnosis\n"
	.size	.L.str.13, 52

	.type	.Lconstinit,@object             # @constinit
	.section	.rodata,"a",@progbits
	.p2align	2, 0x0
.Lconstinit:
	.long	16                              # 0x10
	.long	4096                            # 0x1000
	.long	4096                            # 0x1000
	.long	32                              # 0x20
	.long	4096                            # 0x1000
	.long	4096                            # 0x1000
	.long	64                              # 0x40
	.long	4096                            # 0x1000
	.long	4096                            # 0x1000
	.long	128                             # 0x80
	.long	4096                            # 0x1000
	.long	4096                            # 0x1000
	.long	16                              # 0x10
	.long	1024                            # 0x400
	.long	4096                            # 0x1000
	.long	16                              # 0x10
	.long	2048                            # 0x800
	.long	1024                            # 0x400
	.size	.Lconstinit, 72

	.type	.L.str.14,@object               # @.str.14
	.section	.rodata.str1.1,"aMS",@progbits,1
.L.str.14:
	.asciz	"=== Plain optimized (direct16 vs direct32) median of 20 reps (300 iters each) ===\n"
	.size	.L.str.14, 83

	.type	.L.str.15,@object               # @.str.15
.L.str.15:
	.asciz	"M"
	.size	.L.str.15, 2

	.type	.L.str.16,@object               # @.str.16
.L.str.16:
	.asciz	" plain16 median "
	.size	.L.str.16, 17

	.type	.L.str.17,@object               # @.str.17
.L.str.17:
	.asciz	"ms (min "
	.size	.L.str.17, 9

	.type	.L.str.18,@object               # @.str.18
.L.str.18:
	.asciz	" max "
	.size	.L.str.18, 6

	.type	.L.str.19,@object               # @.str.19
.L.str.19:
	.asciz	") | plain32 median "
	.size	.L.str.19, 20

	.type	.L.str.20,@object               # @.str.20
.L.str.20:
	.asciz	"ms\n"
	.size	.L.str.20, 4

	.type	.L.str.21,@object               # @.str.21
.L.str.21:
	.asciz	"\n=== M5 baseline fused_m5_k32 vs plain (median per shape, interleaved) ===\n"
	.size	.L.str.21, 76

	.type	.L.str.22,@object               # @.str.22
.L.str.22:
	.asciz	" plain median "
	.size	.L.str.22, 15

	.type	.L.str.23,@object               # @.str.23
.L.str.23:
	.asciz	" ms_fused_m5 median "
	.size	.L.str.23, 21

	.type	.L.str.24,@object               # @.str.24
.L.str.24:
	.asciz	" ratio median "
	.size	.L.str.24, 15

	.type	.L.str.25,@object               # @.str.25
.L.str.25:
	.asciz	"SLOW"
	.size	.L.str.25, 5

	.type	.L.str.26,@object               # @.str.26
.L.str.26:
	.asciz	"\n=== M6 repacked_k32 (coalesced) vs plain (median) ===\n"
	.size	.L.str.26, 56

	.type	.L.str.27,@object               # @.str.27
.L.str.27:
	.asciz	"ms TF "
	.size	.L.str.27, 7

	.type	.L.str.28,@object               # @.str.28
.L.str.28:
	.asciz	" BW "
	.size	.L.str.28, 5

	.type	.L.str.29,@object               # @.str.29
.L.str.29:
	.asciz	" | m6_repacked median "
	.size	.L.str.29, 23

	.type	.L.str.30,@object               # @.str.30
.L.str.30:
	.asciz	" effBW "
	.size	.L.str.30, 8

	.type	.L.str.31,@object               # @.str.31
.L.str.31:
	.asciz	" ratio "
	.size	.L.str.31, 8

	.type	.L.str.32,@object               # @.str.32
.L.str.32:
	.asciz	"\n=== M6 repacked_db_k32 (double-buffer) median ===\n"
	.size	.L.str.32, 52

	.type	.L.str.33,@object               # @.str.33
.L.str.33:
	.asciz	" m6_db median "
	.size	.L.str.33, 15

	.type	.L.str.34,@object               # @.str.34
.L.str.34:
	.asciz	"\n=== M6 repacked_k64 (16x16x64) median ===\n"
	.size	.L.str.34, 44

	.type	.L.str.35,@object               # @.str.35
.L.str.35:
	.asciz	" m6_k64 median "
	.size	.L.str.35, 16

	.type	.L.str.36,@object               # @.str.36
.L.str.36:
	.asciz	"\n=== Isolated decode probe (N=4096 K=4096 k32, 8 MB packed ->16 MB decoded) ===\n"
	.size	.L.str.36, 81

	.type	.L.str.37,@object               # @.str.37
.L.str.37:
	.asciz	"scattered decode "
	.size	.L.str.37, 18

	.type	.L.str.38,@object               # @.str.38
.L.str.38:
	.asciz	"ms decoded BW "
	.size	.L.str.38, 15

	.type	.L.str.39,@object               # @.str.39
.L.str.39:
	.asciz	" GB/s\n"
	.size	.L.str.39, 7

	.type	.L.str.40,@object               # @.str.40
.L.str.40:
	.asciz	"repacked coalesced decode "
	.size	.L.str.40, 27

	.type	.L.str.41,@object               # @.str.41
.L.str.41:
	.asciz	"ms packed BW "
	.size	.L.str.41, 14

	.type	.L.str.42,@object               # @.str.42
.L.str.42:
	.asciz	" decoded BW "
	.size	.L.str.42, 13

	.type	.L.str.43,@object               # @.str.43
.L.str.43:
	.asciz	"plain saving at "
	.size	.L.str.43, 17

	.type	.L.str.44,@object               # @.str.44
.L.str.44:
	.asciz	" GB/s = "
	.size	.L.str.44, 9

	.type	.L.str.45,@object               # @.str.45
.L.str.45:
	.asciz	" ms; repacked decode "
	.size	.L.str.45, 22

	.type	.L.str.46,@object               # @.str.46
.L.str.46:
	.asciz	" ms "
	.size	.L.str.46, 5

	.type	.L.str.47,@object               # @.str.47
.L.str.47:
	.asciz	"> saving (cannot hide)"
	.size	.L.str.47, 23

	.type	.L.str.48,@object               # @.str.48
.L.str.48:
	.asciz	"< saving"
	.size	.L.str.48, 9

	.type	.L.str.49,@object               # @.str.49
.L.str.49:
	.asciz	"\nRepack host cost (once per weight, not per GEMM): "
	.size	.L.str.49, 52

	.type	.L.str.50,@object               # @.str.50
.L.str.50:
	.asciz	" ms for N="
	.size	.L.str.50, 11

	.type	.L.str.51,@object               # @.str.51
.L.str.51:
	.asciz	" K="
	.size	.L.str.51, 4

	.type	.L.str.52,@object               # @.str.52
.L.str.52:
	.asciz	" ("
	.size	.L.str.52, 3

	.type	.L.str.53,@object               # @.str.53
.L.str.53:
	.asciz	" MB) BW "
	.size	.L.str.53, 9

	.type	.L.str.54,@object               # @.str.54
.L.str.54:
	.asciz	"Repack amortizes: weight is stationary, repack is offline preprocessing (legal per M6 spec), not counted in GEMM time.\n"
	.size	.L.str.54, 120

	.type	.L.str.55,@object               # @.str.55
.L.str.55:
	.asciz	"cannot create std::vector larger than max_size()"
	.size	.L.str.55, 49

	.type	.L.str.56,@object               # @.str.56
.L.str.56:
	.asciz	"vector::_M_default_append"
	.size	.L.str.56, 26

	.type	.L.str.57,@object               # @.str.57
.L.str.57:
	.asciz	"vector::_M_realloc_append"
	.size	.L.str.57, 26

	.type	.L__unnamed_1,@object           # @0
.L__unnamed_1:
	.asciz	"_Z14plain_direct16PK14__hip_fp8_e4m3S1_Pfiii"
	.size	.L__unnamed_1, 45

	.type	.L__unnamed_2,@object           # @1
.L__unnamed_2:
	.asciz	"_Z14plain_direct32PK14__hip_fp8_e4m3S1_Pfiii"
	.size	.L__unnamed_2, 45

	.type	.L__unnamed_3,@object           # @2
.L__unnamed_3:
	.asciz	"_Z12fused_m5_k32PK14__hip_fp8_e4m3PKhS3_Pfiiii"
	.size	.L__unnamed_3, 47

	.type	.L__unnamed_4,@object           # @3
.L__unnamed_4:
	.asciz	"_Z21fused_m6_repacked_k32PK14__hip_fp8_e4m3PKhS3_Pfiii"
	.size	.L__unnamed_4, 55

	.type	.L__unnamed_5,@object           # @4
.L__unnamed_5:
	.asciz	"_Z24fused_m6_repacked_db_k32PK14__hip_fp8_e4m3PKhS3_Pfiii"
	.size	.L__unnamed_5, 58

	.type	.L__unnamed_6,@object           # @5
.L__unnamed_6:
	.asciz	"_Z21fused_m6_repacked_k64PK14__hip_fp8_e4m3PKhS3_Pfiii"
	.size	.L__unnamed_6, 55

	.type	.L__unnamed_7,@object           # @6
.L__unnamed_7:
	.asciz	"_Z21decode_only_scatteredPKhS0_Phiii"
	.size	.L__unnamed_7, 37

	.type	.L__unnamed_8,@object           # @7
.L__unnamed_8:
	.asciz	"_Z20decode_only_repackedPKhS0_Phii"
	.size	.L__unnamed_8, 35

	.type	.L__unnamed_9,@object           # @8
	.section	.hip_fatbin,"a",@progbits
	.p2align	12, 0x0
.L__unnamed_9:
	.asciz	"__CLANG_OFFLOAD_BUNDLE__\002\000\000\000\000\000\000\000\000\020\000\000\000\000\000\000\000\000\000\000\000\000\000\000\036\000\000\000\000\000\000\000host-x86_64-unknown-linux-gnu-\000\020\000\000\000\000\000\000\300b\000\000\000\000\000\000 \000\000\000\000\000\000\000hipv4-amdgcn-amd-amdhsa--gfx1201\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\177ELF\002\001\001@\004\000\000\000\000\000\000\000\003\000\340\000\001\000\000\000\000\000\000\000\000\000\000\000@\000\000\000\000\000\000\000\200^\000\000\000\000\000\000N\000\000\000@\0008\000\t\000@\000\021\000\017\000\006\000\000\000\004\000\000\000@\000\000\000\000\000\000\000@\000\000\000\000\000\000\000@\000\000\000\000\000\000\000\370\001\000\000\000\000\000\000\370\001\000\000\000\000\000\000\b\000\000\000\000\000\000\000\001\000\000\000\004\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000@)\000\000\000\000\000\000@)\000\000\000\000\000\000\000\020\000\000\000\000\000\000\001\000\000\000\005\000\000\000\000*\000\000\000\000\000\000\000:\000\000\000\000\000\000\000:\000\000\000\000\000\000\200,\000\000\000\000\000\000\200,\000\000\000\000\000\000\000\020\000\000\000\000\000\000\001\000\000\000\006\000\000\000\200V\000\000\000\000\000\000\200v\000\000\000\000\000\000\200v\000\000\000\000\000\000p\000\000\000\000\000\000\000\200\t\000\000\000\000\000\000\000\020\000\000\000\000\000\000\001\000\000\000\006\000\000\000\360V\000\000\000\000\000\000\360\206\000\000\000\000\000\000\360\206\000\000\000\000\000\000\000\000\000\000\000\000\000\000\001\000\000\000\000\000\000\000\000\020\000\000\000\000\000\000\002\000\000\000\006\000\000\000\200V\000\000\000\000\000\000\200v\000\000\000\000\000\000\200v\000\000\000\000\000\000p\000\000\000\000\000\000\000p\000\000\000\000\000\000\000\b\000\000\000\000\000\000\000R\345td\004\000\000\000\200V\000\000\000\000\000\000\200v\000\000\000\000\000\000\200v\000\000\000\000\000\000p\000\000\000\000\000\000\000\200\t\000\000\000\000\000\000\001\000\000\000\000\000\000\000Q\345td\006\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\004\000\000\000\004\000\000\0008\002\000\000\000\000\000\0008\002\000\000\000\000\000\0008\002\000\000\000\000\000\000\324\036\000\000\000\000\000\000\324\036\000\000\000\000\000\000\004\000\000\000\000\000\000\000\007\000\000\000\300\036\000\000 \000\000\000AMDGPU\000\000\203\256amdhsa.kernels\230\336\000\023\245.args\226\204\256.address_space\246global\247.offset\000\245.size\b\253.value_kind\255global_buffer\204\256.address_space\246global\247.offset\b\245.size\b\253.value_kind\255global_buffer\204\256.address_space\246global\247.offset\020\245.size\b\253.value_kind\255global_buffer\203\247.offset\030\245.size\004\253.value_kind\250by_value\203\247.offset\034\245.size\004\253.value_kind\250by_value\203\247.offset \245.size\004\253.value_kind\250by_value\261.gfx1250_revision\242B0\271.group_segment_fixed_size\000\266.kernarg_segment_align\b\265.kernarg_segment_size$\251.language\250OpenCL C\261.language_version\222\002\000\270.max_flat_workgroup_size\315\004\000\245.name\331,_Z14plain_direct16PK14__hip_fp8_e4m3S1_Pfiii\273.private_segment_fixed_size\000\253.sgpr_count\016\261.sgpr_spill_count\000\247.symbol\331/_Z14plain_direct16PK14__hip_fp8_e4m3S1_Pfiii.kd\270.uniform_work_group_size\001\263.uses_dynamic_stack\302\253.vgpr_count\022\261.vgpr_spill_count\000\257.wavefront_size \271.workgroup_processor_mode\001\336\000\023\245.args\226\204\256.address_space\246global\247.offset\000\245.size\b\253.value_kind\255global_buffer\204\256.address_space\246global\247.offset\b\245.size\b\253.value_kind\255global_buffer\204\256.address_space\246global\247.offset\020\245.size\b\253.value_kind\255global_buffer\203\247.offset\030\245.size\004\253.value_kind\250by_value\203\247.offset\034\245.size\004\253.value_kind\250by_value\203\247.offset \245.size\004\253.value_kind\250by_value\261.gfx1250_revision\242B0\271.group_segment_fixed_size\000\266.kernarg_segment_align\b\265.kernarg_segment_size$\251.language\250OpenCL C\261.language_version\222\002\000\270.max_flat_workgroup_size\315\004\000\245.name\331,_Z14plain_direct32PK14__hip_fp8_e4m3S1_Pfiii\273.private_segment_fixed_size\000\253.sgpr_count\016\261.sgpr_spill_count\000\247.symbol\331/_Z14plain_direct32PK14__hip_fp8_e4m3S1_Pfiii.kd\270.uniform_work_group_size\001\263.uses_dynamic_stack\302\253.vgpr_count\026\261.vgpr_spill_count\000\257.wavefront_size \271.workgroup_processor_mode\001\336\000\023\245.args\230\204\256.address_space\246global\247.offset\000\245.size\b\253.value_kind\255global_buffer\204\256.address_space\246global\247.offset\b\245.size\b\253.value_kind\255global_buffer\204\256.address_space\246global\247.offset\020\245.size\b\253.value_kind\255global_buffer\204\256.address_space\246global\247.offset\030\245.size\b\253.value_kind\255global_buffer\203\247.offset \245.size\004\253.value_kind\250by_value\203\247.offset$\245.size\004\253.value_kind\250by_value\203\247.offset(\245.size\004\253.value_kind\250by_value\203\247.offset,\245.size\004\253.value_kind\250by_value\261.gfx1250_revision\242B0\271.group_segment_fixed_size\315\f\000\266.kernarg_segment_align\b\265.kernarg_segment_size0\251.language\250OpenCL C\261.language_version\222\002\000\270.max_flat_workgroup_size\315\004\000\245.name\331._Z12fused_m5_k32PK14__hip_fp8_e4m3PKhS3_Pfiiii\273.private_segment_fixed_size\000\253.sgpr_count\021\261.sgpr_spill_count\000\247.symbol\3311_Z12fused_m5_k32PK14__hip_fp8_e4m3PKhS3_Pfiiii.kd\270.uniform_work_group_size\001\263.uses_dynamic_stack\302\253.vgpr_count\034\261.vgpr_spill_count\000\257.wavefront_size \271.workgroup_processor_mode\001\336\000\023\245.args\227\204\256.address_space\246global\247.offset\000\245.size\b\253.value_kind\255global_buffer\204\256.address_space\246global\247.offset\b\245.size\b\253.value_kind\255global_buffer\204\256.address_space\246global\247.offset\020\245.size\b\253.value_kind\255global_buffer\204\256.address_space\246global\247.offset\030\245.size\b\253.value_kind\255global_buffer\203\247.offset \245.size\004\253.value_kind\250by_value\203\247.offset$\245.size\004\253.value_kind\250by_value\203\247.offset(\245.size\004\253.value_kind\250by_value\261.gfx1250_revision\242B0\271.group_segment_fixed_size\315\013\000\266.kernarg_segment_align\b\265.kernarg_segment_size,\251.language\250OpenCL C\261.language_version\222\002\000\270.max_flat_workgroup_size\315\004\000\245.name\3316_Z21fused_m6_repacked_k32PK14__hip_fp8_e4m3PKhS3_Pfiii\273.private_segment_fixed_size\000\253.sgpr_count\016\261.sgpr_spill_count\000\247.symbol\3319_Z21fused_m6_repacked_k32PK14__hip_fp8_e4m3PKhS3_Pfiii.kd\270.uniform_work_group_size\001\263.uses_dynamic_stack\302\253.vgpr_count\034\261.vgpr_spill_count\000\257.wavefront_size \271.workgroup_processor_mode\001\336\000\023\245.args\227\204\256.address_space\246global\247.offset\000\245.size\b\253.value_kind\255global_buffer\204\256.address_space\246global\247.offset\b\245.size\b\253.value_kind\255global_buffer\204\256.address_space\246global\247.offset\020\245.size\b\253.value_kind\255global_buffer\204\256.address_space\246global\247.offset\030\245.size\b\253.value_kind\255global_buffer\203\247.offset \245.size\004\253.value_kind\250by_value\203\247.offset$\245.size\004\253.value_kind\250by_value\203\247.offset(\245.size\004\253.value_kind\250by_value\261.gfx1250_revision\242B0\271.group_segment_fixed_size\315\016\000\266.kernarg_segment_align\b\265.kernarg_segment_size,\251.language\250OpenCL C\261.language_version\222\002\000\270.max_flat_workgroup_size\315\004\000\245.name\3319_Z24fused_m6_repacked_db_k32PK14__hip_fp8_e4m3PKhS3_Pfiii\273.private_segment_fixed_size\000\253.sgpr_count\017\261.sgpr_spill_count\000\247.symbol\331<_Z24fused_m6_repacked_db_k32PK14__hip_fp8_e4m3PKhS3_Pfiii.kd\270.uniform_work_group_size\001\263.uses_dynamic_stack\302\253.vgpr_count\036\261.vgpr_spill_count\000\257.wavefront_size \271.workgroup_processor_mode\001\336\000\023\245.args\227\204\256.address_space\246global\247.offset\000\245.size\b\253.value_kind\255global_buffer\204\256.address_space\246global\247.offset\b\245.size\b\253.value_kind\255global_buffer\204\256.address_space\246global\247.offset\020\245.size\b\253.value_kind\255global_buffer\204\256.address_space\246global\247.offset\030\245.size\b\253.value_kind\255global_buffer\203\247.offset \245.size\004\253.value_kind\250by_value\203\247.offset$\245.size\004\253.value_kind\250by_value\203\247.offset(\245.size\004\253.value_kind\250by_value\261.gfx1250_revision\242B0\271.group_segment_fixed_size\315\016\000\266.kernarg_segment_align\b\265.kernarg_segment_size,\251.language\250OpenCL C\261.language_version\222\002\000\270.max_flat_workgroup_size\315\004\000\245.name\3316_Z21fused_m6_repacked_k64PK14__hip_fp8_e4m3PKhS3_Pfiii\273.private_segment_fixed_size\000\253.sgpr_count\016\261.sgpr_spill_count\000\247.symbol\3319_Z21fused_m6_repacked_k64PK14__hip_fp8_e4m3PKhS3_Pfiii.kd\270.uniform_work_group_size\001\263.uses_dynamic_stack\302\253.vgpr_count*\261.vgpr_spill_count\000\257.wavefront_size \271.workgroup_processor_mode\001\336\000\023\245.args\334\000\023\204\256.address_space\246global\247.offset\000\245.size\b\253.value_kind\255global_buffer\204\256.address_space\246global\247.offset\b\245.size\b\253.value_kind\255global_buffer\204\256.address_space\246global\247.offset\020\245.size\b\253.value_kind\255global_buffer\203\247.offset\030\245.size\004\253.value_kind\250by_value\203\247.offset\034\245.size\004\253.value_kind\250by_value\203\247.offset \245.size\004\253.value_kind\250by_value\203\247.offset(\245.size\004\253.value_kind\264hidden_block_count_x\203\247.offset,\245.size\004\253.value_kind\264hidden_block_count_y\203\247.offset0\245.size\004\253.value_kind\264hidden_block_count_z\203\247.offset4\245.size\002\253.value_kind\263hidden_group_size_x\203\247.offset6\245.size\002\253.value_kind\263hidden_group_size_y\203\247.offset8\245.size\002\253.value_kind\263hidden_group_size_z\203\247.offset:\245.size\002\253.value_kind\262hidden_remainder_x\203\247.offset<\245.size\002\253.value_kind\262hidden_remainder_y\203\247.offset>\245.size\002\253.value_kind\262hidden_remainder_z\203\247.offsetP\245.size\b\253.value_kind\266hidden_global_offset_x\203\247.offsetX\245.size\b\253.value_kind\266hidden_global_offset_y\203\247.offset`\245.size\b\253.value_kind\266hidden_global_offset_z\203\247.offseth\245.size\002\253.value_kind\260hidden_grid_dims\261.gfx1250_revision\242B0\271.group_segment_fixed_size\000\266.kernarg_segment_align\b\265.kernarg_segment_size\315\001(\251.language\250OpenCL C\261.language_version\222\002\000\270.max_flat_workgroup_size\315\004\000\245.name\331$_Z21decode_only_scatteredPKhS0_Phiii\273.private_segment_fixed_size\000\253.sgpr_count\016\261.sgpr_spill_count\000\247.symbol\331'_Z21decode_only_scatteredPKhS0_Phiii.kd\270.uniform_work_group_size\001\263.uses_dynamic_stack\302\253.vgpr_count\006\261.vgpr_spill_count\000\257.wavefront_size \271.workgroup_processor_mode\001\336\000\023\245.args\334\000\022\204\256.address_space\246global\247.offset\000\245.size\b\253.value_kind\255global_buffer\204\256.address_space\246global\247.offset\b\245.size\b\253.value_kind\255global_buffer\204\256.address_space\246global\247.offset\020\245.size\b\253.value_kind\255global_buffer\203\247.offset\030\245.size\004\253.value_kind\250by_value\203\247.offset\034\245.size\004\253.value_kind\250by_value\203\247.offset \245.size\004\253.value_kind\264hidden_block_count_x\203\247.offset$\245.size\004\253.value_kind\264hidden_block_count_y\203\247.offset(\245.size\004\253.value_kind\264hidden_block_count_z\203\247.offset,\245.size\002\253.value_kind\263hidden_group_size_x\203\247.offset.\245.size\002\253.value_kind\263hidden_group_size_y\203\247.offset0\245.size\002\253.value_kind\263hidden_group_size_z\203\247.offset2\245.size\002\253.value_kind\262hidden_remainder_x\203\247.offset4\245.size\002\253.value_kind\262hidden_remainder_y\203\247.offset6\245.size\002\253.value_kind\262hidden_remainder_z\203\247.offsetH\245.size\b\253.value_kind\266hidden_global_offset_x\203\247.offsetP\245.size\b\253.value_kind\266hidden_global_offset_y\203\247.offsetX\245.size\b\253.value_kind\266hidden_global_offset_z\203\247.offset`\245.size\002\253.value_kind\260hidden_grid_dims\261.gfx1250_revision\242B0\271.group_segment_fixed_size\000\266.kernarg_segment_align\b\265.kernarg_segment_size\315\001 \251.language\250OpenCL C\261.language_version\222\002\000\270.max_flat_workgroup_size\315\004\000\245.name\331\"_Z20decode_only_repackedPKhS0_Phii\273.private_segment_fixed_size\000\253.sgpr_count\n\261.sgpr_spill_count\000\247.symbol\331%_Z20decode_only_repackedPKhS0_Phii.kd\270.uniform_work_group_size\001\263.uses_dynamic_stack\302\253.vgpr_count\006\261.vgpr_spill_count\000\257.wavefront_size \271.workgroup_processor_mode\001\255amdhsa.target\272amdgcn-amd-amdhsa--gfx1201\256amdhsa.version\222\001\002\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\001\000\000\000\022\003\007\000\000:\000\000\000\000\000\000\340\002\000\000\000\000\000\000\034\001\000\000\022\003\007\000\000H\000\000\000\000\000\000\020\006\000\000\000\000\000\000\215\001\000\000\022\003\007\000\000O\000\000\000\000\000\000\330\006\000\000\000\000\000\000u\002\000\000\022\003\007\000\000_\000\000\000\000\000\000\000\003\000\000\000\000\000\000.\000\000\000\021\003\006\000@'\000\000\000\000\000\000@\000\000\000\000\000\000\000S\001\000\000\021\003\006\000\000(\000\000\000\000\000\000@\000\000\000\000\000\000\000\307\001\000\000\021\003\006\000@(\000\000\000\000\000\000@\000\000\000\000\000\000\000\004\002\000\000\022\003\007\000\000V\000\000\000\000\000\0000\b\000\000\000\000\000\000\232\002\000\000\021\003\006\000\300(\000\000\000\000\000\000@\000\000\000\000\000\000\000^\000\000\000\022\003\007\000\000=\000\000\000\000\000\000\030\003\000\000\000\000\000\000\273\000\000\000\022\003\007\000\000A\000\000\000\000\000\000\244\006\000\000\000\000\000\000;\002\000\000\021\003\006\000\200(\000\000\000\000\000\000@\000\000\000\000\000\000\000\302\002\000\000\022\003\007\000\000b\000\000\000\000\000\000\204\002\000\000\000\000\000\000\013\003\000\000\021\000\n\000\360\206\000\000\000\000\000\000\001\000\000\000\000\000\000\000\213\000\000\000\021\003\006\000\200'\000\000\000\000\000\000@\000\000\000\000\000\000\000\352\000\000\000\021\003\006\000\300'\000\000\000\000\000\000@\000\000\000\000\000\000\000\345\002\000\000\021\003\006\000\000)\000\000\000\000\000\000@\000\000\000\000\000\000\000\004\000\000\000\001\000\000\000\004\000\000\000\032\000\000\000 \000\000\000\000\002\000\000\002Be\020\200. \000\006@\001\000@\b\b\202\030\024\001\000\020\200\000@\001\000\000\000\005\000\000\000\n\000\000\000\017\000\000\000\220\027\345\347\344\223\265\017\314FN(]\324\236\325l\177\261&@\213\256@(5a\024h\2726\251\271\230Q\315N\2718\237\216R6\376\246\337\256\013R\257\277W\377\r\273Aj\260\325Y\252\301\273\007\357\027\203\023\022\000\000\000\022\000\000\000\006\000\000\000\016\000\000\000\007\000\000\000\b\000\000\000\000\000\000\000\r\000\000\000\000\000\000\000\013\000\000\000\000\000\000\000\n\000\000\000\000\000\000\000\000\000\000\000\021\000\000\000\003\000\000\000\t\000\000\000\000\000\000\000\020\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\004\000\000\000\000\000\000\000\000\000\000\000\002\000\000\000\000\000\000\000\001\000\000\000\000\000\000\000\f\000\000\000\017\000\000\000\005\000\000\000\000_Z14plain_direct16PK14__hip_fp8_e4m3S1_Pfiii\000_Z14plain_direct16PK14__hip_fp8_e4m3S1_Pfiii.kd\000_Z14plain_direct32PK14__hip_fp8_e4m3S1_Pfiii\000_Z14plain_direct32PK14__hip_fp8_e4m3S1_Pfiii.kd\000_Z12fused_m5_k32PK14__hip_fp8_e4m3PKhS3_Pfiiii\000_Z12fused_m5_k32PK14__hip_fp8_e4m3PKhS3_Pfiiii.kd\000_Z21fused_m6_repacked_k32PK14__hip_fp8_e4m3PKhS3_Pfiii\000_Z21fused_m6_repacked_k32PK14__hip_fp8_e4m3PKhS3_Pfiii.kd\000_Z24fused_m6_repacked_db_k32PK14__hip_fp8_e4m3PKhS3_Pfiii\000_Z24fused_m6_repacked_db_k32PK14__hip_fp8_e4m3PKhS3_Pfiii.kd\000_Z21fused_m6_repacked_k64PK14__hip_fp8_e4m3PKhS3_Pfiii\000_Z21fused_m6_repacked_k64PK14__hip_fp8_e4m3PKhS3_Pfiii.kd\000_Z21decode_only_scatteredPKhS0_Phiii\000_Z21decode_only_scatteredPKhS0_Phiii.kd\000_Z20decode_only_repackedPKhS0_Phii\000_Z20decode_only_repackedPKhS0_Phii.kd\000__hip_cuid_437800c2e6994a55\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000$\000\000\000\000\000\000\000\300\022\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000`\000\000\000\002\000\017\340\204\001\000\000\b\004\000\000\000\000\000\000\000\000\000\000\000\000\000\000$\000\000\000\000\000\000\000\200\025\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000p\000\000\000\002\000\017\340\204\001\000\000\b\004\000\000\000\000\000\000\000\f\000\000\000\000\000\0000\000\000\000\000\000\000\000@\031\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\340\000\000\000\003\000\017\340\204\001\000\000\b\004\000\000\000\000\000\000\000\013\000\000\000\000\000\000,\000\000\000\000\000\000\000\000 \000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\320\000\000\000\003\000\017\340\204\001\000\000\b\004\000\000\000\000\000\000\000\016\000\000\000\000\000\000,\000\000\000\000\000\000\000\300&\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\340\000\000\000\003\000\017\340\204\001\000\000\b\004\000\000\000\000\000\000\000\016\000\000\000\000\000\000,\000\000\000\000\000\000\000\200-\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\020\001\000\000\005\000\017\340\204\001\000\000\b\004\000\000\000\000\000\000\000\000\000\000\000\000\000\000(\001\000\000\000\000\000\000@6\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000`\000\000\000\000\000\017\340\204\000\000\000\b\004\000\000\000\000\000\000\000\000\000\000\000\000\000\000 \001\000\000\000\000\000\000\0009\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000`\000\000\000\000\000\017\340\204\000\000\000\b\004\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\002\000\205\277\200 \000\364\034\000\000\370\000A\000\364\000\000\000\370\000 \000\364\020\000\000\370\217\000\0246\201\000\0262\200\000\210\276\000\000\307\277\003\200\002\277\003\000\242\277\217\000\0226\210\026\0206\001\000\240\277\301\000\210\276\200\000\020\312\200\000\006\007\200\000\020\312\200\000\004\005\200\000\020\312\200\000\002\003\200\000\020\312\200\000\000\001~\bj\2217\000\244\277\200\002\000~\210\026\0206s\003\b\226u\003\t\226\b\204\b\204\t\204\n\204\001|\376\326\003\024\"\004\b\237\t\206\n\237\013\206\004\b\204\251\006\n\206\251\000\001\020\312\000\001\002\002\000\001\020\312\000\001\004\004\t\004\000\327\004\002\002\002\237\361\210\277\013| \325\005\000\021\000\f\004\000\327\006\002\002\002\237\361\210\277\r| \325\007\000\021\000\000\001\020\312\000\001\006\001\000\003\016~\200\001\204\276\236\377\210\277\016j\000\327\t\t\000\002\235\377\210\277\017| \325\005\026\252\001\020j\000\327\f\t\000\002\235\377\210\277\021| \325\005\032\252\001|@\005\356\016\000\000\000\016\000\000\000|@\005\356\020\000\000\000\020\000\000\000\004\220\204\251\236\377\210\277\004\003\003\277\000\000\300\277\000@F\314\016!\002\034\350\377\241\277\n\003\022~\001\000\207\277\b|\376\326\002\020&\004\200\002\022~s\002\003\226u\204\006\204\236\377\210\277\003\204\004\204\006\237\007\206\236\377\210\277\004\237\005\206\006\202\206\204\202\020\024>\002\020\020J\236\377\210\277\004\202\204\204\236\377\210\277\000\004\200\251\202\020\030>\002\020\020J\000\006\200\251\031\001\207\277\nj\000\327\000\024\002\002\202\020\034>\002\020\020J\235\377\210\277\013| \325\001\026\252\001\fj\000\327\000\030\002\002\235\377\210\277\r| \325\001\032\252\001\016j\000\327\000\034\002\002\202\020 >\002\020\020J\235\377\210\277\017| \325\001\036\252\001\002\000\205\277|\200\006\356\000\000\000\000\n\000\000\000|\200\006\356\000\000\200\000\f\000\000\000|\200\006\356\000\000\000\001\016\000\000\000\202\020\000>\002\020\020J\nj\000\327\000 \002\002\235\377\210\277\013| \325\001\"\252\001\323\001\207\277\202\020\030>\002\020\020J\000j\000\327\000\000\002\002\235\377\210\277\001| \325\001\002\252\001\202\020\034>\002\020\020J\fj\000\327\000\030\002\002\235\377\210\277\r| \325\001\032\252\001\303\001\207\277\202\020\020>\016j\000\327\000\034\002\002\235\377\210\277\017| \325\001\036\252\001\bj\000\327\000\020\002\002\235\377\210\277\t| \325\001\022\252\001\004\000\205\277|\200\006\356\000\000\200\001\n\000\000\000|\200\006\356\000\000\000\002\000\000\000\000|\200\006\356\000\000\200\002\f\000\000\000|\200\006\356\000\000\000\003\016\000\000\000|\200\006\356\000\000\200\003\b\000\000\000\000\000\260\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\002\000\205\277\200 \000\364\034\000\000\370\000A\000\364\000\000\000\370\000 \000\364\020\000\000\370\217\000\0326\200\000\210\276\000\000\307\277\003\200\002\277\002\000\242\277\217\000\0226\001\000\240\277\301\000\210\276\200\000\020\312\200\000\006\b\200\000\020\312\200\000\004\006\200\000\020\312\200\000\002\004\200\000\020\312\200\000\000\002~\bj\221D\000\244\277\220\000\0026u\003\b\226s\003\t\226\b\204\b\204\t\204\n\204\002|\376\326\003\032\006\004\b\237\t\206\n\237\013\206\006\b\206\251\004\n\204\251\200\002\002~\003\006\000\327\006\004\002\002\237\361\210\277\004| \325\007\000\031\000\002\004\000\327\004\004\002\002\237\361\210\277\005| \325\005\000\021\000\tj\000\327\003\021\001\002\001\000\207\277\n| \325\200\b\252\001\013j\000\327\002\021\001\002\235\377\210\277\f| \325\200\n\252\001\001\001\020\312\001\001\002\002\001\001\020\312\001\001\004\004\001\001\020\312\001\001\006\006\001\003\020~\200\000\204\276|\300\005\356\016\000\000\000\013\370\377\377|\300\005\356\022\000\000\000\t\370\377\377\tj\000\327\tA\001\002\235\377\210\277\n| \325\200\024\252\001\013j\000\327\013A\001\002\235\377\210\277\f| \325\200\030\252\001\236\377\210\277\004\240\004\201\236\377\210\277\004\003\003\277\000\000\300\277\001@F\314\016%\006\034\001\000\207\277\001@F\314\020)\006\034\345\377\241\277\r\003\022~\201\000\0002s\002\003\226u\204\006\204\236\377\210\277\003\204\004\204\006\237\007\206\210\000\0006\236\377\210\277\004\237\005\206\006\202\206\204\236\377\210\277\004\202\204\204\236\377\210\277\000\004\200\251\t|\376\326\002\000&\004\200\002\024~\000\006\200\251\241\000\207\277\202\022\026>\002\022\022J\202\022\032>\002\022\022J\264\001\207\277\013j\000\327\000\026\002\002\235\377\210\277\f| \325\001\030\252\001\202\022\036>\002\022\022J\rj\000\327\000\032\002\002\235\377\210\277\016| \325\001\034\252\001\003\000\207\277\202\022\">\002\022\022J\001\000\205\277|\200\006\356\000\000\200\000\013\000\000\000|\200\006\356\000\000\000\001\r\000\000\000\017j\000\327\000\036\002\002\235\377\210\277\020| \325\001 \252\001\202\022\000>\002\022\022J\002j\000\327\000\"\002\002|\200\006\356\000\000\200\001\017\000\000\000\235\377\210\277\003| \325\001$\252\001\202\022\026>\002\022\022J\000j\000\327\000\000\002\002\235\377\210\277\001| \325\001\002\252\001\323\001\207\277\202\022\032>\002\022\022J\013j\000\327\000\026\002\002\235\377\210\277\f| \325\001\030\252\001\202\022\022>\rj\000\327\000\032\002\002\235\377\210\277\016| \325\001\034\252\001\003\000\207\277\tj\000\327\000\022\002\002\235\377\210\277\n| \325\001\024\252\001\004\000\205\277|\200\006\356\000\000\000\002\002\000\000\000|\200\006\356\000\000\200\002\000\000\000\000|\200\006\356\000\000\000\003\013\000\000\000|\200\006\356\000\000\200\003\r\000\000\000|\200\006\356\000\000\000\004\t\000\000\000\000\000\260\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000a\000\364\000\000\000\370~\000\203\276\377\000\230}\000\001\000\000%\000\245\277\203\000\0060\000\000\307\277\221\000\207\277\001\002\000\327\b\006\002\002\002| \325\t\000\t\000\200\000\210\276\021\000\240\277\236\377\210\277~\002~\214\377\006\bJ\000\001\000\000\377\006\222|\377\006\000\000\001\002\000\327\377\002\002\002\000\001\000\000\237\361\210\277\002| \325\200\004\n\000\004\003\006~j\b\b\214\236\377\210\277~\b~\221\013\000\245\277~\000\202\276\377\006\230}\371\007\000\000\353\377\245\277|@\005\356\004\000\000\000\001\000\000\000\000\000\300\277\000\0004\331\003\004\000\000\344\377\240\277~\003~\214\000\243\000\364$\000\000\370\000\000\306\277\301N\200\276\217\000\0326\201\000\0342\200\000\200\276\377\377\224\277|\300\n\356\000\000\004\000\000\000\000\000\000\000\307\277\r\200\002\277\003\000\242\277\217\000\0226\201\000\0242\001\000\240\277\301\000\200\276\200\000\020\312\200\000\006\b\200\000\020\312\200\000\004\006\200\000\020\312\200\000\002\004\200\000\020\312\200\000\000\002~\000j\221u\204\002\204\367\000\244\277\236\377\210\277\002\034\002J\220\000\0046s\r\000\226\205\032\f0\000\204\b\204\f\002\210|\017\000,\327\016\002\002\002\200\002\002~\003|\376\326\r\032\n\004\236\377\210\277\b\237\t\206\204\000\n0\236\377\210\277\004\b\204\251\021\000X\326\006\005\376\003\000\b\000\000\000\000L\324\240\000\002\002\201\000\0006\020\000F\326\016\013\375\003\000\b\000\000\002\001\000\327\004\006\002\002\241\001\207\277\003| \325\005\000\005\000\001\003\b~\t\001\000\327\002\021\001\002\001\003\004~\220\n$6\237\361\210\277\n| \325\200\006\006\000\001\003\006~\001\001\"\312\203\000\000\005\001\003\016~\022\001\020\312\001\001\006\023\001\003\020~\200\000\203\276\200\000\204\276\200\000\205\2761\000\240\277\236\377\210\277~\016~\214A\001\207\277\025\000D\326\0271\376\003\004\000\f\f\000\000\306\277\f\000D\326\026\031\376\003\004\000\f\f\024%(J\f\000V\326\f!U\004\b\0004\331\024\013\000\000\236\377\210\277~\t~\214\000\000\310\277\301N\200\276\013\000F\326\b\022E\004\220\000\000J\240&&J\005\240\005\201\004\201\004\201\003\220\003\201\236\377\210\277\005\r\003\277\377\377\224\277|\300\n\356\000\000\004\000\000\000\000\000|\300\005\356\024\000\000\000\t\370\377\377\000\000\374\333\013\000\000\030\t\001\000\327\tA\001\002\237\361\210\277\n| \325\200\024\006\000\000\000\310\277\001@F\314\0241\006\034\001\000\207\277\001@F\314\0265\006\034\222\000\242\277\236\377\210\277\003\377\001\213\200\377\377?\004\201\b\213\236\377\210\277\001\036*J\024\000F\326\b\022A\004\000 \211\276H\000\245\277\2008\030~\001\000D\324\r&\002\002\200\002\026~\003\000\207\277\f9,~\f90~\f9.~j\001\001\213\236\377\210\277\001 \216\276-\000\245\277\377\000\0266x\000\000\000\221\000\207\277\025\027\026J\237\026\0304\013\001\000\327\006\026\002\002\237\361\210\277\002\000\207\277\f| \325\007\030\006\000|\000\005\356\f\000\000\000\013\000\000\000\000\000\300\277\201\030\0260\217\030,2\207\030.2\227\030\0302\024\002\207\277\377\026\0266\376\001\000\000\377,,6\376\001\000\000\024\002\207\277\377.06\376\001\000\000\377\03028\001\006\000\000\377\03046\376\001\000\000\000\000\230\332\013\000\000\013\000\004\230\332\026\000\000\027\000\000\210\332\031\000\000\f\002\000\306\277\000\002\234\332\030\000\000\013\000\006\210\332\032\000\000\026\003\000\306\277\030\0009\327\210.\002\002\236\377\210\277~\016~\214A\001\207\277\027\000D\326\0271\376\003\004\000\f\f\000\000\306\277\f\000D\326\026\031\376\003\004\000\f\f\024%,J\f\000V\326\f!]\004\000\0004\331\026\013\000\000\236\377\210\277~\t~\214\000 \211\276\210\377\245\277\200\000 \312\210&\026\013\2008\030~\022\001\207\277\001\000D\324\r,\002\002\f9,~\f90~\f9.~j\001\001\213\236\377\210\277\001 \216\276l\377\245\277\204\000\026J\221\000\207\277\377\026\0266|\000\000\000\025\027\026J1\001\207\277\237\026\0304\013\001\000\327\006\026\002\002\237\361\210\277\f| \325\007\030\006\000|\000\005\356\f\000\000\000\013\000\000\000\000\000\300\277\201\030\0260\217\030*2\207\030,2\227\030\0302\024\002\207\277\377\026\0266\376\001\000\000\377**6\376\001\000\000\024\002\207\277\377,,6\376\001\000\000\377\03008\001\006\000\000\377\03026\376\001\000\000\000\000\230\332\013\000\000\013\000\004\230\332\025\000\000\027\000\000\210\332\030\000\000\f\002\000\306\277\000\002\234\332\026\000\000\013\000\006\210\332\031\000\000\026\003\000\306\277\030\0009\327\210.\002\002=\377\240\277\r\001\020\312\016\001\n\t\001\000\207\277\210\024\0006s\f\000\226\236\377\210\277\002\237\003\206\000\204\000\204\236\377\210\277\002\202\202\204\t|\376\326\f\000&\004\200\002\024~\000\237\001\206\236\377\210\277\000\202\200\204\236\377\210\277\n\000\200\251\236\377\210\277\000\002\200\251\202\022\026>\f\022\022J1\002\207\277\202\022\032>\f\022\022J\236\377\210\277\013j\000\327\000\026\002\002\221\001\207\277\f| \325\001\030\252\001\202\022\036>\f\022\022J\rj\000\327\000\032\002\002\235\377\210\277\016| \325\001\034\252\001\003\000\207\277\202\022\">\f\022\022J\001\000\205\277|\200\006\356\000\000\200\000\013\000\000\000|\200\006\356\000\000\000\001\r\000\000\000\017j\000\327\000\036\002\002\235\377\210\277\020| \325\001 \252\001\202\022\000>\f\022\022J\002j\000\327\000\"\002\002|\200\006\356\000\000\200\001\017\000\000\000\235\377\210\277\003| \325\001$\252\001\202\022\026>\f\022\022J\000j\000\327\000\000\002\002\235\377\210\277\001| \325\001\002\252\001\323\001\207\277\202\022\032>\f\022\022J\013j\000\327\000\026\002\002\235\377\210\277\f| \325\001\030\252\001\202\022\022>\rj\000\327\000\032\002\002\235\377\210\277\016| \325\001\034\252\001\003\000\207\277\tj\000\327\000\022\002\002\235\377\210\277\n| \325\001\024\252\001\004\000\205\277|\200\006\356\000\000\000\002\002\000\000\000|\200\006\356\000\000\200\002\000\000\000\000|\200\006\356\000\000\000\003\013\000\000\000|\200\006\356\000\000\200\003\r\000\000\000|\200\006\356\000\000\000\004\t\000\000\000\000\000\260\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000a\000\364\000\000\000\370\203\000\0260~\000\203\276\377\000\230}\000\001\000\000&\000\245\277\203\000\0020\013\003\006~\000\000\307\277\222\000\207\277\001\002\000\327\b\002\002\002\002| \325\t\000\t\000\200\000\210\276\021\000\240\277\236\377\210\277~\002~\214\377\006\bJ\000\001\000\000\377\006\222|\377\006\000\000\001\002\000\327\377\002\002\002\000\001\000\000\237\361\210\277\002| \325\200\004\n\000\004\003\006~j\b\b\214\236\377\210\277~\b~\221\013\000\245\277~\000\202\276\377\006\230}\371\007\000\000\353\377\245\277|@\005\356\004\000\000\000\001\000\000\000\000\000\300\277\000\0004\331\003\004\000\000\344\377\240\277~\003~\214\200 \000\364$\000\000\370\000\000\306\277\301N\200\276\217\000\0306\201\000\0322\200\000\200\276\377\377\224\277|\300\n\356\000\000\004\000\000\000\000\000\000\000\307\277\003\200\002\277\003\000\242\277\217\000\0226\201\000\0242\001\000\240\277\301\000\200\276\200\000\020\312\200\000\006\b\200\000\020\312\200\000\004\006\200\000\020\312\200\000\002\004\200\000\020\312\200\000\000\002~\000j\221\320\000\244\277\201\000\0020\220\000\0046\002\237\000\206s\003\001\226\000\234\000\205\202\002\0166\200\002\002~\003|\376\326\003\030\n\004\004\000V\326\f\013\375\003\000\b\000\000\002\000\b\201\001\204\000\204\005\000F\326\r\t\375\003\000\n\000\000\000\237\001\206\001\001 \312\004\005\022\002\201\016\0208\004\000\200\251\006\000F\326\r\013\375\003\000\b\000\000\003\000\000\327\000\006\002\002\001\001\"\312\202\016\016\004\203\016\0160\202\020\"0\203\020\0200\237\361\210\277\n| \325\001\000\001\000\240\000\230|\000\006\000\327\006\026\002\002\t\000\000\327\003\021\001\002\237\361\210\277\016| \325\007\000\031\000\n| \325\200\024\002\000\005\037\036J\001\001 \312\006\017\020\003\001\001 \312\006\021\022\007\001\003\f~\001\003\020~\005#\"J\001\003\n~\236\377\210\277\b\204\004\204u\210\001\204\004\377\004\213\000\377\377\377\200\000\205\276!\000\240\277\236\377\210\277~\000~\214\000\000\310\277\301N\200\276\005\240\005\201\001\004\001\201\005\003\003\277\377\377\224\277|\300\n\356\000\000\004\000\000\000\000\000|\300\005\356\024\000\000\000\t\370\377\377\000\000\374\333\023\000\000\030\t\000\000\327\tA\001\002\237\361\210\277\n| \325\200\024\002\000\000\000\310\277\001@F\314\0241\006\034\001\000\207\277\001@F\314\0265\006\034\301N\200\276\377\377\224\277|\300\n\356\000\000\004\000\000\000\000\000p\000\242\277j \206\276\r\000\245\277\236\377\210\277\001\237\007\206\024\000\000\327\000\003\000\002\236\361\210\277\025| \325\007\034\002\000|@\005\356\024\000\000\000\024\000\000\000\000\000\300\277\000\n4\331\013\024\000\000\236\377\210\277~\006~\214\000\000\310\277\301N\200\276\377\377\224\277|\300\n\356\000\000\004\000\000\000\000\000j \200\276)\000\245\277\000\000\330\330\017\000\000\024\000\000\306\277\227(*2\201(.0\217(,2\207((24\002\207\277\377*08\001\006\000\000\377**6\376\001\000\000\377..6\376\001\000\000\377((6\376\001\000\000\377,,6\376\001\000\000\000\000\350\330\030\000\000\030\000\006\350\330\025\000\000\025\000\000\360\330\027\000\000\027\000\002\360\330\024\000\000\024\000\004\360\330\026\000\000\026\003\000\306\277\025\000D\326\0251\376\003\004\000\f\f\001\000\306\277\024\000V\326\024!]\004\000\000\306\277\002\000\207\277\025\000V\326\025!Y\004\000\0004\331\020\024\000\000\236\377\210\277~\000~\214j \200\276\231\377\245\277\000\000\330\330\021\000\000\024\000\000\306\277\227(*2\201(.0\217(,2\207((24\002\207\277\377*08\001\006\000\000\377**6\376\001\000\000\377..6\376\001\000\000\377((6\376\001\000\000\377,,6\376\001\000\000\000\000\350\330\030\000\000\030\000\006\350\330\025\000\000\025\000\000\360\330\027\000\000\027\000\002\360\330\024\000\000\024\000\004\360\330\026\000\000\026\003\000\306\277\025\000D\326\0251\376\003\004\000\f\f\001\000\306\277\024\000V\326\024!]\004\000\000\306\277\002\000\207\277\025\000V\326\025!Y\004\000\0004\331\022\024\000\000o\377\240\277\f\001\020\312\r\001\n\t\001\000\207\277\210\024\0006s\002\000\226\236\377\210\277u\204\004\204\000\204\000\204\004\237\005\206\t|\376\326\002\000&\004\200\002\024~\236\377\210\277\000\237\001\206\004\202\204\204\236\377\210\277\000\202\200\204\236\377\210\277\n\000\200\251\236\377\210\277\000\004\200\251\202\022\026>\002\022\022J1\002\207\277\202\022\032>\002\022\022J\236\377\210\277\013j\000\327\000\026\002\002\221\001\207\277\f| \325\001\030\252\001\202\022\036>\002\022\022J\rj\000\327\000\032\002\002\235\377\210\277\016| \325\001\034\252\001\004\000\207\277\017j\000\327\000\036\002\002\202\022\">\002\022\022J\235\377\210\277\020| \325\001 \252\001\002\000\205\277|\200\006\356\000\000\200\000\013\000\000\000|\200\006\356\000\000\000\001\r\000\000\000|\200\006\356\000\000\200\001\017\000\000\000\202\022\000>\002\022\022J\002j\000\327\000\"\002\002\235\377\210\277\003| \325\001$\252\001\323\001\207\277\202\022\026>\002\022\022J\000j\000\327\000\000\002\002\235\377\210\277\001| \325\001\002\252\001\202\022\032>\002\022\022J\013j\000\327\000\026\002\002\235\377\210\277\f| \325\001\030\252\001\303\001\207\277\202\022\022>\rj\000\327\000\032\002\002\235\377\210\277\016| \325\001\034\252\001\tj\000\327\000\022\002\002\235\377\210\277\n| \325\001\024\252\001\004\000\205\277|\200\006\356\000\000\000\002\002\000\000\000|\200\006\356\000\000\200\002\000\000\000\000|\200\006\356\000\000\000\003\013\000\000\000|\200\006\356\000\000\200\003\r\000\000\000|\200\006\356\000\000\000\004\t\000\000\000\000\000\260\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000a\000\364\000\000\000\370\203\000\0240~\000\203\276\377\000\230}\000\001\000\000&\000\245\277\203\000\0020\n\003\006~\000\000\307\277\222\000\207\277\001\002\000\327\b\002\002\002\002| \325\t\000\t\000\200\000\210\276\021\000\240\277\236\377\210\277~\002~\214\377\006\bJ\000\001\000\000\377\006\222|\377\006\000\000\001\002\000\327\377\002\002\002\000\001\000\000\237\361\210\277\002| \325\200\004\n\000\004\003\006~j\b\b\214\236\377\210\277~\b~\221\013\000\245\277~\000\202\276\377\006\230}\371\007\000\000\353\377\245\277|@\005\356\004\000\000\000\001\000\000\000\000\000\300\277\000\0004\331\003\004\000\000\344\377\240\277~\003~\214\200 \000\364$\000\000\370\000\000\306\277\301N\200\276\000\000L\324\240\000\002\002\377\377\224\277|\300\n\356\000\000\004\000\000\000\000\000\000\000\307\277\003\237\002\277\301\200\001\230\t\000\207\277\001\000\b\213\236\377\210\277\b \201\276\013\000\245\277u\210\b\204\236\377\210\277\b\237\t\206\236\377\210\277\006\b\210\251\b@\005\356\001\000\000\000\n\000\000\000\000\000\300\277\000\f4\331\n\001\000\000~\001~\214\000\000\310\277\301N\200\276\217\000\0266\201\000\0302\200\000\201\276\003\200\002\277\377\377\224\277|\300\n\356\000\000\004\000\000\000\000\000\003\000\242\277\217\000\0226\201\000\0322\001\000\240\277\301\000\201\276\200\000\020\312\200\000\006\b\200\000\020\312\200\000\004\006\200\000\020\312\200\000\002\004\200\000\020\312\200\000\000\002~\001j\221\347\000\244\277\201\000\0020\220\000\0006s\003\b\226\002\237\001\206\236\377\210\277\b\204\b\204\202\002\b6\002|\376\326\003\026\002\004\236\377\210\277\b\237\t\206\001\234\001\205\236\377\210\277\004\b\204\251\201\b\0068\200\002\002~\205\026\n0\002\001\001\201\002\004\000\327\004\004\002\002\003\000\207\277\001\001\"\312\202\006\020\006\001\001\"\312\203\006\022\007\237\361\210\277\003| \325\005\000\021\000\001\204\f\206\023\001\000\327\006\024\002\002\tj\000\327\002\021\001\002\r\000F\326\f\t\375\003\000\f\000\000\016\000F\326\f\013\375\003\000\b\000\000\024| \325\007\000\005\000\001\001 \312\377\024\024\b\000\f\000\000\n| \325\200\006\252\001\001\003\004~\000\000X\326\005\001\376\003\000\b\000\000\001\003\006~\202\b\0360\001\001\"\312\203\b\020\005\001\003\b~u\f\001\201\f\210\004\204\236\377\210\277\001\210\001\204\240\000\205\276\200\000\206\276\025\000\240\277\236\377\210\277~\b~\214\000\000\310\277\301N\200\276\tj\000\327\tA\001\002\235\377\210\277\n| \325\200\024\252\001\001\004\001\201\005\240\005\201\006\201\006\201\200\000\210\276\377\377\224\277|\300\n\356\000\000\004\000\000\000\000\000\236\377\210\277~\bj\213\236\377\210\277\222\000\244\277\236\377\210\277\006\201\007\213\236\377\210\277\027\000F\326\007\0205\004\026\000F\326\007\0229\004\000 \210\276+\000\245\277\002\000\207\277\027\0370J\000\000\330\330\030\000\000\030\000\000\306\277\227022\201040\207062\217002\004\000\207\277\377288\001\006\000\000\377226\376\001\000\000\377446\376\001\000\000\377666\376\001\000\000\377006\376\001\000\000\000\000\350\330\034\000\000\034\000\006\350\330\031\000\000\031\000\000\360\330\032\000\000\032\000\002\360\330\033\000\000\033\000\004\360\330\030\000\000\035\003\000\306\277\031\000D\326\0319\376\003\004\000\f\f\026!8J\001\000\306\277\030\000V\326\033!i\004\000\000\306\277\031\000V\326\031!u\004\000\0004\331\034\030\000\000\236\377\210\277~\b~\214\000 \210\276*\000\245\277\027#.J\000\000\330\330\027\000\000\027\000\000\306\277\227.02\201.20\207.42\217..2\004\000\207\277\377068\001\006\000\000\377006\376\001\000\000\377226\376\001\000\000\377446\376\001\000\000\377..6\376\001\000\000\000\000\350\330\033\000\000\033\000\006\350\330\030\000\000\030\000\000\360\330\031\000\000\031\000\002\360\330\032\000\000\032\000\004\360\330\027\000\000\027\003\000\306\277\030\000D\326\0307\376\003\004\000\f\f\026%6J\001\000\306\277\026\000V\326\032!e\004\000\000\306\277\027\000V\326\030!]\004\000\0004\331\033\026\000\000\236\377\210\277~\b~\214\000\000\310\277\301N\200\276\032\000F\326\007\022\001\004\005\003\003\277\301\000\210\276\377\377\224\277|\300\n\356\000\000\004\000\000\000\000\000|\300\005\356\026\000\000\000\t\370\377\377\000\000\374\333\032\000\000\032\000\000\310\277\001@F\314\0265\006\034\001\000\207\277\001@F\314\0309\006\034\301N\200\276\377\377\224\277|\300\n\356\000\000\004\000\000\000\000\000}\377\242\277\000 \210\276j\377\245\277\001\237\t\206\026j\000\327\023\003\000\002\234\377\210\277\027| \325\t(\252\001\007\201\007\215\236\377\210\277\030\000F\326\007\020U\004|@\005\356\026\000\000\000\026\000\000\000\000\000\300\277\000\0004\331\030\026\000\000Y\377\240\277\013\003\022~\f\003\032~\001\000\207\277\210\032\0006s\002\000\226\236\377\210\277u\204\004\204\000\204\000\204\236\377\210\277\004\237\005\206\t|\376\326\002\000&\004\200\002\024~\000\237\001\206\236\377\210\277\004\202\204\204\000\202\200\204\236\377\210\277\n\000\200\251\236\377\210\277\000\004\200\251\202\022\026>\002\022\022J1\002\207\277\202\022\032>\002\022\022J\236\377\210\277\013j\000\327\000\026\002\002\235\377\210\277\f| \325\001\030\252\001\202\022\036>\002\022\022J\rj\000\327\000\032\002\002\235\377\210\277\016| \325\001\034\252\001\004\000\207\277\017j\000\327\000\036\002\002\202\022\">\002\022\022J\235\377\210\277\020| \325\001 \252\001\002\000\205\277|\200\006\356\000\000\200\000\013\000\000\000|\200\006\356\000\000\000\001\r\000\000\000|\200\006\356\000\000\200\001\017\000\000\000\202\022\000>\002\022\022J\002j\000\327\000\"\002\002\235\377\210\277\003| \325\001$\252\001\323\001\207\277\202\022\026>\002\022\022J\000j\000\327\000\000\002\002\235\377\210\277\001| \325\001\002\252\001\202\022\032>\002\022\022J\013j\000\327\000\026\002\002\235\377\210\277\f| \325\001\030\252\001\303\001\207\277\202\022\022>\rj\000\327\000\032\002\002\235\377\210\277\016| \325\001\034\252\001\tj\000\327\000\022\002\002\235\377\210\277\n| \325\001\024\252\001\004\000\205\277|\200\006\356\000\000\000\002\002\000\000\000|\200\006\356\000\000\200\002\000\000\000\000|\200\006\356\000\000\000\003\013\000\000\000|\200\006\356\000\000\200\003\r\000\000\000|\200\006\356\000\000\000\004\t\000\000\000\000\000\260\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000a\000\364\000\000\000\370\203\000\0240~\000\203\276\377\000\230}\000\001\000\000&\000\245\277\203\000\0020\n\003\006~\000\000\307\277\222\000\207\277\001\002\000\327\b\002\002\002\002| \325\t\000\t\000\200\000\210\276\021\000\240\277\236\377\210\277~\002~\214\377\006\bJ\000\001\000\000\377\006\222|\377\006\000\000\001\002\000\327\377\002\002\002\000\001\000\000\237\361\210\277\002| \325\200\004\n\000\004\003\006~j\b\b\214\236\377\210\277~\b~\221\013\000\245\277~\000\202\276\377\006\230}\371\007\000\000\353\377\245\277|@\005\356\004\000\000\000\001\000\000\000\000\000\300\277\000\0004\331\003\004\000\000\344\377\240\277~\003~\214\200 \000\364$\000\000\370\000\000\306\277\301N\200\276\217\000\0266\201\000\0302\200\000\200\276\377\377\224\277|\300\n\356\000\000\004\000\000\000\000\000\000\000\307\277\003\200\002\277\003\000\242\277\217\000\0226\201\000\0322\001\000\240\277\301\000\200\276\200\000\020\312\200\000\006\b\200\000\020\312\200\000\004\006\200\000\020\312\200\000\002\004\200\000\020\312\200\000\000\002~\000j\221X\001\244\277\201\000\0020\002\237\000\206s\003\001\226\000\234\000\205\300\000\230|\200\000$\312\240\002\002\001\002\000\b\201\001\204\000\204\206\000\0160\002\000\207\277\003|\376\326\003\026\n\004\000\237\001\206\202\000\n0\004\000\200\251\300\003\004\260\006\000F\326\f\013\375\003\000\f\000\000\004\000F\326\f\r\375\003\000\b\000\000\r\000\000\327\000\006\002\002\003\000W\326\007\t\374\003\000\b\000\000\237\361\210\277\016| \325\001\000\001\000\000\000L\324\240\000\002\002\017\001\000\327\006\024\002\002\001\001 \312\003\005\030\002\204\n\n6\237\361\210\277\020| \325\007\000\005\000\236\377\210\277\b\205\001\204u\211\004\204\201\n\0008\202\n\0160\202\n\0208\203\n\0220\203\n\n8\202\000&0\001\001\"\312\203\000\024\003\202\020*0\203\020\0200\202\n.0\203\n\n0\377\024\000J\000\377\377\377\004\023$J\006'&J\004)(J\006+*J\006/.J\001\001 \312\004\013\030\005\004\021,J\001\003\b~\001\001 \312\006\017\020\b\001\001\020\312\001\001\006\006\236\377\210\277\001\377\005\213\000\376\377\377\200\000\206\276.\000\240\277\236\377\210\277~\001~\214\000\000\310\277\301N\200\276\t\001\000\327\r\r\000\002\237\361\210\277\n| \325\200\034\006\000\006\300\006\201\004\005\004\201\236\377\210\277\006\003\003\277\377\377\224\277|\300\n\356\000\000\004\000\000\000\000\000\001\000\205\277|\300\005\356\032\000\000\000\t\020\000\000|\300\005\356\036\000\000\000\t\000\000\000\000\000\374\333\031\000\000\"\020\000\374\333\031\000\000&\001\000\310\277\001@F\314\036E\006\034\241\000\207\277\001@F\314 I\006\034\000\000\306\277\001@F\314\032M\006\034\001\000\207\277\001@F\314\034Q\006\034\301N\200\276\377\377\224\277|\300\n\356\000\000\004\000\000\000\000\000\334\000\242\277j \207\276\037\000\245\277\004\237\b\206\t\001\000\327\017\t\000\002\236\361\210\277\n| \325\b \006\000\000\0034~\200\000\210\276|@\005\356\033\000\000\000\t\000\000\000\t\001\000\327\377\022\002\002\000\001\000\000\237\361\210\277\n| \325\200\024\006\000\000\000\300\277\000\r4\331\032\033\000\000\032\t\000\327\3774\002\002\000\001\000\000\t\301\t\215\236\377\210\277~\t\001\213\236\377\210\277\001\b\b\214\236\377\210\277~\b~\221\351\377\246\277\236\377\210\277~\007~\214\000\000\310\277\301N\200\276\377\377\224\277|\300\n\356\000\000\004\000\000\000\000\000\000 \201\276)\000\245\277\000\000\330\330\021\000\000\t\000\000\306\277\227\022\0242\201\02240\207\02262\217\022\0222\004\000\207\277\377\02488\001\006\000\000\377\024\0246\376\001\000\000\377446\376\001\000\000\377666\376\001\000\000\377\022\0226\376\001\000\000\000\000\350\330\034\000\000\034\000\006\350\330\n\000\000\n\000\000\360\330\032\000\000\032\000\002\360\330\033\000\000\033\000\004\360\330\t\000\000\035\003\000\306\277\n\000D\326\n9\376\003\004\000\f\f\001\000\306\277\t\000V\326\033!i\004\000\000\306\277\002\000\207\277\n\000V\326\n!u\004\000\0004\331\022\t\000\000\236\377\210\277~\001~\214\000 \201\276)\000\245\277\000\000\330\330\023\000\000\t\000\000\306\277\227\022\0242\201\02240\207\02262\217\022\0222\004\000\207\277\377\02488\001\006\000\000\377\024\0246\376\001\000\000\377446\376\001\000\000\377666\376\001\000\000\377\022\0226\376\001\000\000\000\000\350\330\034\000\000\034\000\006\350\330\n\000\000\n\000\000\360\330\032\000\000\032\000\002\360\330\033\000\000\033\000\004\360\330\t\000\000\035\003\000\306\277\n\000D\326\n9\376\003\004\000\f\f\001\000\306\277\t\000V\326\033!i\004\000\000\306\277\002\000\207\277\n\000V\326\n!u\004\000\0004\331\024\t\000\000\236\377\210\277~\001~\214\000 \201\276)\000\245\277\000\000\330\330\025\000\000\t\000\000\306\277\227\022\0242\201\02240\207\02262\217\022\0222\004\000\207\277\377\02488\001\006\000\000\377\024\0246\376\001\000\000\377446\376\001\000\000\377666\376\001\000\000\377\022\0226\376\001\000\000\000\000\350\330\034\000\000\034\000\006\350\330\n\000\000\n\000\000\360\330\032\000\000\032\000\002\360\330\033\000\000\033\000\004\360\330\t\000\000\035\003\000\306\277\n\000D\326\n9\376\003\004\000\f\f\001\000\306\277\t\000V\326\033!i\004\000\000\306\277\002\000\207\277\n\000V\326\n!u\004\000\0004\331\026\t\000\000\236\377\210\277~\001~\214\000 \201\276 \377\245\277\000\000\330\330\027\000\000\t\000\000\306\277\227\022\0242\201\02240\207\02262\217\022\0222\004\000\207\277\377\02488\001\006\000\000\377\024\0246\376\001\000\000\377446\376\001\000\000\377666\376\001\000\000\377\022\0226\376\001\000\000\000\000\350\330\034\000\000\034\000\006\350\330\n\000\000\n\000\000\360\330\032\000\000\032\000\002\360\330\033\000\000\033\000\004\360\330\t\000\000\035\003\000\306\277\n\000D\326\n9\376\003\004\000\f\f\001\000\306\277\t\000V\326\033!i\004\000\000\306\277\002\000\207\277\n\000V\326\n!u\004\000\0004\331\030\t\000\000\366\376\240\277\013\003\022~\f\003\032~\001\000\207\277\210\032\0006s\002\000\226\236\377\210\277u\204\004\204\000\204\000\204\236\377\210\277\004\237\005\206\t|\376\326\002\000&\004\200\002\024~\000\237\001\206\236\377\210\277\004\202\204\204\000\202\200\204\236\377\210\277\n\000\200\251\236\377\210\277\000\004\200\251\202\022\026>\002\022\022J1\002\207\277\202\022\032>\002\022\022J\236\377\210\277\013j\000\327\000\026\002\002\221\001\207\277\f| \325\001\030\252\001\202\022\036>\002\022\022J\rj\000\327\000\032\002\002\235\377\210\277\016| \325\001\034\252\001\004\000\207\277\017j\000\327\000\036\002\002\202\022\">\002\022\022J\235\377\210\277\020| \325\001 \252\001\002\000\205\277|\200\006\356\000\000\200\000\013\000\000\000|\200\006\356\000\000\000\001\r\000\000\000|\200\006\356\000\000\200\001\017\000\000\000\202\022\000>\002\022\022J\002j\000\327\000\"\002\002\235\377\210\277\003| \325\001$\252\001\323\001\207\277\202\022\026>\002\022\022J\000j\000\327\000\000\002\002\235\377\210\277\001| \325\001\002\252\001\202\022\032>\002\022\022J\013j\000\327\000\026\002\002\235\377\210\277\f| \325\001\030\252\001\303\001\207\277\202\022\022>\rj\000\327\000\032\002\002\235\377\210\277\016| \325\001\034\252\001\tj\000\327\000\022\002\002\235\377\210\277\n| \325\001\024\252\001\004\000\205\277|\200\006\356\000\000\000\002\002\000\000\000|\200\006\356\000\000\200\002\000\000\000\000|\200\006\356\000\000\000\003\013\000\000\000|\200\006\356\000\000\200\003\r\000\000\000|\200\006\356\000\000\000\004\t\000\000\000\000\000\260\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\001\000\205\277\200\000\000\3644\000\000\370\000\241\000\364\030\000\000\370\000\000\307\277\002\377\002\213\377\377\000\000\005\237\003\206\000|\376\326u\004\000\004\003\235\002\205\236\377\210\277\005\002\002\201\236\377\210\277\002\203\002\206\236\377\210\277\002\004\003\226\236\377\210\277\003\000\210|j \203\276\251\000\245\277\002\025\203\276\000B\000\364\000\000\000\370\003e\204\276\200\003\205\201\000 \000\364\020\000\000\370\231\002\207\277\004V\002~\001\005\b~\200\000\002L\004\377\004\242\376\377\177O\241\004\207\277\001\001\002$\236\377\210\277\004g\204\276\236\377\210\277\n\000\207\277\005\004\005\226\236\377\210\277\004\005\205\226\236\377\210\277\004\005\004\201\236\377\210\277\002\000-\327\001\t\000\002\221\000\207\277\003\000,\327\002\007\000\002\001\007\002L\201\004\006J\"\001\207\277\003\002\bN\003\002\226|\002\007R\312\001\t\000\002\002\000\006:\222\001\207\277\201\004\bJ\003\002\226|\243\001\207\277\237\006\0064\235\377\210\277\002\t\002\002\221\000\207\277\001\007\002:\001\007\002L!\001\207\277\002\000,\327\001\005\000\002\001\000,\327\001\r\000\002\000\005\004L\203\000\0000\262\001\207\277\003\000\021\326\0029\005\002\203\004\b0\237\004\n4\230\006\0062\022\001\207\277\233\n\n2\004\007\006J\022\001\207\277\002\000G\326\002\013\n\002\377\006\0066\000\377\377\377\022\001\207\277\377\004\0046\200\377\377\377\004\007\006L\221\000\207\277\201\006\0064\001\000U\326\002\003\016\004\237\000\0064B\001\207\277\237\002\0044\000\000\307\277\001j\000\327\b\002\002\002\235\377\210\277\002| \325\t\004\252\001|\000\005\356\004\000\000\000\001\000\000\000\002j\000\327\000\000\002\002\235\377\210\277\003| \325\001\006\252\001\000\000\300\277\201\b\0020\001\000\207\277\377\002\n6\376\001\000\000\n\200\007\356\001\000\000\000\005\000\000\000\000\000\300\277|\000\006\356\000\000\200\000\002\000\000\000\n\200\007\356\000\000\000\000\005\001\000\000\207\b\0022\001\000\207\277\377\002\0026\376\001\000\000\000\000\300\277|\000\006\356\000\000\000\000\002\001\000\000\n\200\007\356\000\000\000\000\001\000\002\000\000\000\300\277|\000\006\356\000\000\000\000\002\002\000\000\n\200\007\356\000\000\000\000\001\001\002\000\217\b\0022\001\000\207\277\377\002\0026\376\001\000\000\000\000\300\277|\000\006\356\000\000\000\000\002\003\000\000\n\200\007\356\000\000\000\000\001\000\004\000\000\000\300\277|\000\006\356\000\000\000\000\002\004\000\000\n\200\007\356\000\000\000\000\001\001\004\000\227\b\0022\001\000\207\277\377\002\b6\376\001\000\000\377\002\0028\001\006\000\000\000\000\300\277|\000\006\356\000\000\000\000\002\005\000\000\n\200\007\356\000\000\000\000\004\000\006\000\000\000\300\277|\000\006\356\000\000\000\000\002\006\000\000\n\200\007\356\000\000\000\000\001\000\000\000\000\000\300\277|\000\006\356\000\000\000\000\002\007\000\000\000\000\260\277\001\000\205\277\000\001\000\364,\000\000\370\200 \000\364\030\000\000\370\000\000\307\277\004\377\004\213\377\377\000\000\003\237\005\206\000|\376\326u\b\000\004\005\235\004\205\236\377\210\277\003\004\003\201\231\004\207\277\003\203\003\206\003\002\003\226~\000\202\276~\000\304\324\003\000\002\002\213\000\245\277\237\000\0024\001\000\205\277\000A\000\364\000\000\000\370\000 \000\364\020\000\000\370\221\000\207\277\232\002\0022\000\003\004J!\001\207\277\377\004\0026\300\377\000\000\202\004\0040\000\003\006L\242\001\207\277\377\004\0046\000\377\377\377\203\000\0000\001\000\021\326\003\001!\002\221\000\207\277\001\0009\327\215\002\002\002\001\000b\327\001\007\001\002\221\000\207\277\004\000\003\327\003\003\002\002\001\000\021\326\004\001!\002\001@b\327\377\b\002\002\374\000\000\000\022\001\207\277\001\000:\327\202\002\002\002\003\020\004\327\003\003\002\002\022\001\207\277\001\000\021\326\001\001A\002\003\000\021\326\003\001!\002\222\000\207\277\001\000F\326\001\t\t\004\001\000F\326\003\005\005\004\237\000\0064\262\000\207\277\237\002\0044\000\000\307\277\001j\000\327\004\002\002\002\002| \325\005\004\252\001|\000\005\356\004\000\000\000\001\000\000\000\002j\000\327\000\000\002\002\235\377\210\277\003| \325\001\006\252\001\000\000\300\277\201\b\0020\001\000\207\277\377\002\n6\376\001\000\000\006\200\007\356\001\000\000\000\005\000\000\000\000\000\300\277|\000\006\356\000\000\200\000\002\000\000\000\006\200\007\356\000\000\000\000\005\001\000\000\207\b\0022\001\000\207\277\377\002\0026\376\001\000\000\000\000\300\277|\000\006\356\000\000\000\000\002\001\000\000\006\200\007\356\000\000\000\000\001\000\002\000\000\000\300\277|\000\006\356\000\000\000\000\002\002\000\000\006\200\007\356\000\000\000\000\001\001\002\000\217\b\0022\001\000\207\277\377\002\0026\376\001\000\000\000\000\300\277|\000\006\356\000\000\000\000\002\003\000\000\006\200\007\356\000\000\000\000\001\000\004\000\000\000\300\277|\000\006\356\000\000\000\000\002\004\000\000\006\200\007\356\000\000\000\000\001\001\004\000\227\b\0022\001\000\207\277\377\002\b6\376\001\000\000\377\002\0028\001\006\000\000\000\000\300\277|\000\006\356\000\000\000\000\002\005\000\000\006\200\007\356\000\000\000\000\004\000\006\000\000\000\300\277|\000\006\356\000\000\000\000\002\006\000\000\006\200\007\356\000\000\000\000\001\000\000\000\000\000\300\277|\000\006\356\000\000\000\000\002\007\000\000\000\000\260\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\006\000\000\000\000\000\000\000\020!\000\000\000\000\000\000\013\000\000\000\000\000\000\000\030\000\000\000\000\000\000\000\005\000\000\000\000\000\000\000\334#\000\000\000\000\000\000\n\000\000\000\000\000\000\000'\003\000\000\000\000\000\000\365\376\377o\000\000\000\000\300\"\000\000\000\000\000\000\004\000\000\000\000\000\000\000D#\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000Linker: AMD LLD 23.0.0 (https://github.com/ROCm/llvm-project.git 46fcb339fb61119b337f973c7ca9e710a319fdd0+PATCHED:440716f8b87be9d8e20ed910e10e5b6d14d57cf6)\000AMD clang version 23.0.0git (https://github.com/ROCm/llvm-project.git 46fcb339fb61119b337f973c7ca9e710a319fdd0+PATCHED:440716f8b87be9d8e20ed910e10e5b6d14d57cf6)\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\001\000\000\000\000\000\361\377\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\025\000\000\000\000\000\361\377\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000)\000\000\000\000\000\361\377\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000=\000\000\000\000\000\361\377\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\200\003\000\000\000\002\b\000\200v\000\000\000\000\000\000\000\000\000\000\000\000\000\000Z\000\000\000\022\003\007\000\000:\000\000\000\000\000\000\340\002\000\000\000\000\000\000\207\000\000\000\021\003\006\000@'\000\000\000\000\000\000@\000\000\000\000\000\000\000\267\000\000\000\022\003\007\000\000=\000\000\000\000\000\000\030\003\000\000\000\000\000\000\344\000\000\000\021\003\006\000\200'\000\000\000\000\000\000@\000\000\000\000\000\000\000\024\001\000\000\022\003\007\000\000A\000\000\000\000\000\000\244\006\000\000\000\000\000\000C\001\000\000\021\003\006\000\300'\000\000\000\000\000\000@\000\000\000\000\000\000\000u\001\000\000\022\003\007\000\000H\000\000\000\000\000\000\020\006\000\000\000\000\000\000\254\001\000\000\021\003\006\000\000(\000\000\000\000\000\000@\000\000\000\000\000\000\000\346\001\000\000\022\003\007\000\000O\000\000\000\000\000\000\330\006\000\000\000\000\000\000 \002\000\000\021\003\006\000@(\000\000\000\000\000\000@\000\000\000\000\000\000\000]\002\000\000\022\003\007\000\000V\000\000\000\000\000\0000\b\000\000\000\000\000\000\224\002\000\000\021\003\006\000\200(\000\000\000\000\000\000@\000\000\000\000\000\000\000\316\002\000\000\022\003\007\000\000_\000\000\000\000\000\000\000\003\000\000\000\000\000\000\363\002\000\000\021\003\006\000\300(\000\000\000\000\000\000@\000\000\000\000\000\000\000\033\003\000\000\022\003\007\000\000b\000\000\000\000\000\000\204\002\000\000\000\000\000\000>\003\000\000\021\003\006\000\000)\000\000\000\000\000\000@\000\000\000\000\000\000\000d\003\000\000\021\000\n\000\360\206\000\000\000\000\000\000\001\000\000\000\000\000\000\000\000.note\000.dynsym\000.gnu.hash\000.hash\000.dynstr\000.rodata\000.text\000.dynamic\000.relro_padding\000.bss\000.AMDGPU.csdata\000.AMDGPU.gpr_maximums\000.comment\000.symtab\000.shstrtab\000.strtab\000\000amdgpu.max_num_vgpr\000amdgpu.max_num_agpr\000amdgpu.max_num_sgpr\000amdgpu.max_num_named_barrier\000_Z14plain_direct16PK14__hip_fp8_e4m3S1_Pfiii\000_Z14plain_direct16PK14__hip_fp8_e4m3S1_Pfiii.kd\000_Z14plain_direct32PK14__hip_fp8_e4m3S1_Pfiii\000_Z14plain_direct32PK14__hip_fp8_e4m3S1_Pfiii.kd\000_Z12fused_m5_k32PK14__hip_fp8_e4m3PKhS3_Pfiiii\000_Z12fused_m5_k32PK14__hip_fp8_e4m3PKhS3_Pfiiii.kd\000_Z21fused_m6_repacked_k32PK14__hip_fp8_e4m3PKhS3_Pfiii\000_Z21fused_m6_repacked_k32PK14__hip_fp8_e4m3PKhS3_Pfiii.kd\000_Z24fused_m6_repacked_db_k32PK14__hip_fp8_e4m3PKhS3_Pfiii\000_Z24fused_m6_repacked_db_k32PK14__hip_fp8_e4m3PKhS3_Pfiii.kd\000_Z21fused_m6_repacked_k64PK14__hip_fp8_e4m3PKhS3_Pfiii\000_Z21fused_m6_repacked_k64PK14__hip_fp8_e4m3PKhS3_Pfiii.kd\000_Z21decode_only_scatteredPKhS0_Phiii\000_Z21decode_only_scatteredPKhS0_Phiii.kd\000_Z20decode_only_repackedPKhS0_Phii\000_Z20decode_only_repackedPKhS0_Phii.kd\000__hip_cuid_437800c2e6994a55\000_DYNAMIC\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\001\000\000\000\007\000\000\000\002\000\000\000\000\000\000\0008\002\000\000\000\000\000\0008\002\000\000\000\000\000\000\324\036\000\000\000\000\000\000\000\000\000\000\000\000\000\000\004\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\007\000\000\000\013\000\000\000\002\000\000\000\000\000\000\000\020!\000\000\000\000\000\000\020!\000\000\000\000\000\000\260\001\000\000\000\000\000\000\005\000\000\000\001\000\000\000\b\000\000\000\000\000\000\000\030\000\000\000\000\000\000\000\017\000\000\000\366\377\377o\002\000\000\000\000\000\000\000\300\"\000\000\000\000\000\000\300\"\000\000\000\000\000\000\204\000\000\000\000\000\000\000\002\000\000\000\000\000\000\000\b\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\031\000\000\000\005\000\000\000\002\000\000\000\000\000\000\000D#\000\000\000\000\000\000D#\000\000\000\000\000\000\230\000\000\000\000\000\000\000\002\000\000\000\000\000\000\000\004\000\000\000\000\000\000\000\004\000\000\000\000\000\000\000\037\000\000\000\003\000\000\000\002\000\000\000\000\000\000\000\334#\000\000\000\000\000\000\334#\000\000\000\000\000\000'\003\000\000\000\000\000\000\000\000\000\000\000\000\000\000\001\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000'\000\000\000\001\000\000\000\002\000\000\000\000\000\000\000@'\000\000\000\000\000\000@'\000\000\000\000\000\000\000\002\000\000\000\000\000\000\000\000\000\000\000\000\000\000@\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000/\000\000\000\001\000\000\000\006\000\000\000\000\000\000\000\000:\000\000\000\000\000\000\000*\000\000\000\000\000\000\200,\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\001\000\000\000\000\000\000\000\000\000\000\000\000\000\0005\000\000\000\006\000\000\000\003\000\000\000\000\000\000\000\200v\000\000\000\000\000\000\200V\000\000\000\000\000\000p\000\000\000\000\000\000\000\005\000\000\000\000\000\000\000\b\000\000\000\000\000\000\000\020\000\000\000\000\000\000\000>\000\000\000\b\000\000\000\003\000\000\000\000\000\000\000\360v\000\000\000\000\000\000\360V\000\000\000\000\000\000\020\t\000\000\000\000\000\000\000\000\000\000\000\000\000\000\001\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000M\000\000\000\b\000\000\000\003\000\000\000\000\000\000\000\360\206\000\000\000\000\000\000\360V\000\000\000\000\000\000\001\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\001\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000R\000\000\000\001\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\360V\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\001\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000a\000\000\000\001\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\360V\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\001\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000v\000\000\000\001\000\000\0000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\360V\000\000\000\000\000\000>\001\000\000\000\000\000\000\000\000\000\000\000\000\000\000\001\000\000\000\000\000\000\000\001\000\000\000\000\000\000\000\177\000\000\000\002\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\0000X\000\000\000\000\000\000(\002\000\000\000\000\000\000\020\000\000\000\006\000\000\000\b\000\000\000\000\000\000\000\030\000\000\000\000\000\000\000\207\000\000\000\003\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000XZ\000\000\000\000\000\000\231\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\001\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\221\000\000\000\003\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\361Z\000\000\000\000\000\000\211\003\000\000\000\000\000\000\000\000\000\000\000\000\000\000\001\000\000\000\000\000\000\000\000\000\000\000\000\000\000"
	.size	.L__unnamed_9, 29376

	.type	__hip_fatbin_wrapper,@object    # @__hip_fatbin_wrapper
	.section	.hipFatBinSegment,"aw",@progbits
	.p2align	3, 0x0
__hip_fatbin_wrapper:
	.long	1212764230                      # 0x48495046
	.long	1                               # 0x1
	.quad	.L__unnamed_9
	.quad	0
	.size	__hip_fatbin_wrapper, 24

	.type	__hip_gpubin_handle_437800c2e6994a55,@object # @__hip_gpubin_handle_437800c2e6994a55
	.local	__hip_gpubin_handle_437800c2e6994a55
	.comm	__hip_gpubin_handle_437800c2e6994a55,8,8
	.section	.init_array,"aw",@init_array
	.p2align	3, 0x0
	.quad	__hip_module_ctor
	.type	__hip_cuid_437800c2e6994a55,@object # @__hip_cuid_437800c2e6994a55
	.bss
	.globl	__hip_cuid_437800c2e6994a55
__hip_cuid_437800c2e6994a55:
	.byte	0                               # 0x0
	.size	__hip_cuid_437800c2e6994a55, 1

	.hidden	DW.ref.__gxx_personality_v0
	.weak	DW.ref.__gxx_personality_v0
	.section	.data.DW.ref.__gxx_personality_v0,"awG",@progbits,DW.ref.__gxx_personality_v0,comdat
	.p2align	3, 0x0
	.type	DW.ref.__gxx_personality_v0,@object
	.size	DW.ref.__gxx_personality_v0, 8
DW.ref.__gxx_personality_v0:
	.quad	__gxx_personality_v0
	.ident	"AMD clang version 23.0.0git (https://github.com/ROCm/llvm-project.git 46fcb339fb61119b337f973c7ca9e710a319fdd0+PATCHED:440716f8b87be9d8e20ed910e10e5b6d14d57cf6)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym _Z29__device_stub__plain_direct16PK14__hip_fp8_e4m3S1_Pfiii
	.addrsig_sym _Z29__device_stub__plain_direct32PK14__hip_fp8_e4m3S1_Pfiii
	.addrsig_sym _Z27__device_stub__fused_m5_k32PK14__hip_fp8_e4m3PKhS3_Pfiiii
	.addrsig_sym _Z36__device_stub__fused_m6_repacked_k32PK14__hip_fp8_e4m3PKhS3_Pfiii
	.addrsig_sym _Z39__device_stub__fused_m6_repacked_db_k32PK14__hip_fp8_e4m3PKhS3_Pfiii
	.addrsig_sym _Z36__device_stub__fused_m6_repacked_k64PK14__hip_fp8_e4m3PKhS3_Pfiii
	.addrsig_sym __gxx_personality_v0
	.addrsig_sym _Z36__device_stub__decode_only_scatteredPKhS0_Phiii
	.addrsig_sym _Z35__device_stub__decode_only_repackedPKhS0_Phii
	.addrsig_sym __hip_module_ctor
	.addrsig_sym __hip_module_dtor
	.addrsig_sym _Unwind_Resume
	.addrsig_sym _Z14plain_direct16PK14__hip_fp8_e4m3S1_Pfiii
	.addrsig_sym _Z14plain_direct32PK14__hip_fp8_e4m3S1_Pfiii
	.addrsig_sym _Z12fused_m5_k32PK14__hip_fp8_e4m3PKhS3_Pfiiii
	.addrsig_sym _Z21fused_m6_repacked_k32PK14__hip_fp8_e4m3PKhS3_Pfiii
	.addrsig_sym _Z24fused_m6_repacked_db_k32PK14__hip_fp8_e4m3PKhS3_Pfiii
	.addrsig_sym _Z21fused_m6_repacked_k64PK14__hip_fp8_e4m3PKhS3_Pfiii
	.addrsig_sym _Z21decode_only_scatteredPKhS0_Phiii
	.addrsig_sym _Z20decode_only_repackedPKhS0_Phii
	.addrsig_sym _ZSt4cout
	.addrsig_sym .L__unnamed_9
	.addrsig_sym __hip_fatbin_wrapper
	.addrsig_sym __hip_cuid_437800c2e6994a55
