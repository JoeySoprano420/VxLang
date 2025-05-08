; -------------------------------------------
; VACU x64 AOT Compiler Scaffold (VxForge Core)
; Target: Windows x64
; Assembler: NASM
; Link with: link.exe /subsystem:console
; -------------------------------------------

BITS 64
GLOBAL main
EXTERN GetStdHandle, WriteConsoleA, ExitProcess, CreateFileA, WriteFile, CloseHandle

SECTION .data
    end_scope_marker db "; end_chamber", 0
    conflict_msg db "; ERROR: function redeclared", 0
    scoped_prefix db "chamber_core_", 0
    user_func_label db "user_func:", 0
    resolve_label db ".resolve_label:", 0
    expr_cmp db "cmp rbx, 1", 0
    expr_jump db "jle .resolve_label", 0
    scoped_label db "; scope: chamber.core", 0
    label_instr db "main:", 0
    func_instr db "push rbp", 0
    arg_instr db "; args parsed", 0
    pulse_instr db "call main", 0
    chamber_instr db "; chamber start", 0
    probe_instr db "cmp rax, 1", 0
    resolve_instr db "mov rax, 1", 0
    macro_instr db "; begin macro expansion", 0
    rigid_instr db "; rigid inline block", 0
    newline db 0x0D, 0x0A, 0
    msg_init db "VxLANG Compiler Booted...", 0
    msg_emit db "[EMIT] Instruction: ", 0
    token_pulse db "pulse", 0
    token_chamber db "chamber", 0
    token_probe db "probe", 0
    token_resolve db "resolve", 0
    token_macro db "@macro", 0
    token_rigid db "@rigid", 0

    input_script db "@macro pulseDef { pulse main() { probe (x <= 1) resolve x } }", 0
    token_buffer db 256 dup(0)
    console_handle dq 0
    bytes_written dq 0

    ast_nodes dq 128 dup(0)
    ast_index dq 0

    ir_stack dq 256 dup(0)
    ir_index dq 0

    out_filename db "generated.asm", 0
    out_handle dq 0

SECTION .bss
    label_counter resq 1
    identifier_buffer resb 64
    scope_table resq 64             ; Symbol table for hashed scope entries
    scope_depth resq 1
    scope_stack resq 16             ; Nested scope stack for chamber hierarchies
    scope_stack_ptr resq 1          ; Pointer to top of scope stack
    input_ptr resq 1
    ch resb 1
    label_counter resq 1
    identifier_buffer resb 64
    scope_table resq 64             ; Symbol table for hashed scope entries
    scope_depth resq 1
    input_ptr resq 1
    ch resb 1
    label_counter resq 1
    identifier_buffer resb 64
    input_ptr resq 1
    ch resb 1

SECTION .text

; Capture user-defined identifiers
capture_identifier:
    mov rsi, token_buffer
    mov rdi, identifier_buffer
.capture:
    mov al, [rsi]
    cmp al, 0
    je .done
    mov [rdi], al
    inc rsi
    inc rdi
    jmp .capture
.done:
    mov byte [rdi], 0
    ret

; Build expression tree (mocked for x <= 1)
build_expression_tree:
    lea rdx, [expr_cmp]
    call write_line
    lea rdx, [expr_jump]
    call write_line
    ret

; Parse and simulate arguments (placeholder for future expression parsing)
parse_arguments:
    lea rdx, [arg_instr]
    call write_line
    ret

; Emit parsed arguments
emit_arguments:
    lea rdx, [arg_instr]
    call write_line
    ret

main:
    call init_console
    call print_init

    lea rax, [input_script]
    mov [input_ptr], rax

    call parse_tokens
    call generate_ir
    call open_output_file
    call emit_instructions
    call close_output_file

    mov ecx, 0
    call ExitProcess

init_console:
    mov ecx, -11
    call GetStdHandle
    mov [console_handle], rax
    ret

print_init:
    mov rcx, [console_handle]
    lea rdx, [msg_init]
    mov r8, strlen(msg_init)
    lea r9, [bytes_written]
    sub rsp, 32
    call WriteConsoleA
    add rsp, 32
    ret

strlen:
    xor rax, rax
.next:
    cmp byte [rdi + rax], 0
    je .done
    inc rax
    jmp .next
.done:
    ret

parse_tokens:
    xor rbx, rbx              ; Reset dynamic label index
.next_token:
    call skip_whitespace
    call extract_token
    cmp byte [token_buffer], 0
    je .done
    call capture_identifier
    call match_token
    jmp .next_token
.done:
    ret

skip_whitespace:
    mov rsi, [input_ptr]
.skip:
    mov al, [rsi]
    cmp al, ' '
    jne .save
    inc rsi
    jmp .skip
.save:
    mov [input_ptr], rsi
    ret

extract_token:
    mov rsi, [input_ptr]
    lea rdi, [token_buffer]
    xor rcx, rcx
.copy:
    mov al, [rsi]
    cmp al, 0
    je .null_term
    cmp al, ' '
    je .null_term
    cmp al, '{'
    je .null_term
    cmp al, '}'
    je .null_term
    mov [rdi + rcx], al
    inc rcx
    inc rsi
    jmp .copy
.null_term:
    mov byte [rdi + rcx], 0
    mov [input_ptr], rsi
    ret

match_token:
    lea rdi, [token_buffer]

    lea rsi, [token_pulse]
    call strcmp
    cmp eax, 0
    je .pulse

    lea rsi, [token_chamber]
    call strcmp
    cmp eax, 0
    je .chamber

    lea rsi, [token_probe]
    call strcmp
    cmp eax, 0
    je .probe

    lea rsi, [token_resolve]
    call strcmp
    cmp eax, 0
    je .resolve

    lea rsi, [token_macro]
    call strcmp
    cmp eax, 0
    je .macro

    lea rsi, [token_rigid]
    call strcmp
    cmp eax, 0
    je .rigid

    ret
.pulse:    mov rax, 1  ; pulse
    jmp store_ast_node
.chamber:  mov rax, 2  ; chamber
    jmp store_ast_node
.probe:    mov rax, 3  ; probe
    jmp store_ast_node
.resolve:  mov rax, 4  ; resolve
    jmp store_ast_node
.macro:    mov rax, 10 ; @macro
    jmp store_ast_node
.rigid:    mov rax, 11 ; @rigid
    jmp store_ast_node

store_ast_node:
    ; Detect scoped function redefinition by hashing identifier and scope
    xor rax, rax
    lea rsi, [identifier_buffer]
.hash_loop:
    mov al, [rsi]
    test al, al
    je .scope_hash
    add rax, rax
    add rax, al
    inc rsi
    jmp .hash_loop
.scope_hash:
    ; Incorporate scope stack top to make fully scoped
    mov rbx, [scope_stack_ptr]
    cmp rbx, 0
    je .hash_done
    dec rbx
    lea rdi, [scope_stack]
    add rdi, rbx
    xor rcx, [rdi]
    add rax, rcx
.hash_done:
    ; Calculate table index for scope
    mov rcx, [scope_depth]
    lea rdi, [scope_table]
    add rdi, rcx
    cmp [rdi], rax
    je .conflict
    mov [rdi], rax
    mov [rdi], rax

    ; Push scope hash to stack
    mov rbx, [scope_stack_ptr]
    lea rdi, [scope_stack]
    mov [rdi + rbx*8], rax
    inc rbx
    mov [scope_stack_ptr], rbx

    ; Store node
    mov rcx, [ast_index]
    mov rbx, ast_nodes
    mov [rbx + rcx*8], rax
    inc rcx
    mov [ast_index], rcx

    ; Handle scoped identifiers
    cmp rax, 2 ; chamber
    jne .check_args
    lea rdx, [scoped_label]
    call write_line
    inc qword [scope_depth]

.check_args:
    cmp rax, 1 ; pulse
    jne .end
    call parse_arguments
.end:
    ret

.conflict:
    lea rdx, [conflict_msg]
    call write_line
    ret

.conflict:
    lea rdx, [conflict_msg]
    call write_line
    ret

generate_ir:
    xor rcx, rcx
.next:
    cmp rcx, [ast_index]
    jge .done
    mov rax, ast_nodes
    mov rdx, [rax + rcx*8]
    mov rbx, [ir_index]
    mov rax, ir_stack
    mov [rax + rbx*8], rdx
    inc rbx
    mov [ir_index], rbx
    inc rcx
    jmp .next
.done:
    ret

open_output_file:
    ; Simulate opening file (stub only)
    ret

close_output_file:
    ; Simulate closing file (stub only)
    ret

emit_instructions:
    xor rcx, 0
.loop:
    cmp rcx, [ir_index]
    jge .done
    mov rbx, ir_stack
    mov rax, [rbx + rcx*8]

    cmp rax, 1
    je emit_pulse
    cmp rax, 2
    je emit_chamber
    cmp rax, 3
    je emit_probe
    cmp rax, 4
    je emit_resolve
    cmp rax, 10
    je emit_macro
    cmp rax, 11
    je emit_rigid

    inc rcx
    jmp .loop
.done:
    ret

emit_pulse:
    ; Emit dynamic label using captured identifier
    lea rsi, [identifier_buffer]
    call write_line_dynamic_label
    call write_line
    lea rdx, [pulse_instr]
    call write_line
    lea rdx, [func_instr]
    call write_line
    call emit_arguments
    ret

emit_chamber:
    ; Emit section/namespace
    lea rdx, [chamber_instr]
    call write_line
    ; Scope entry already handled on parsing
    ret

emit_probe:
    ; Emit cmp/jle conditional
    call build_expression_tree
    lea rdx, [probe_instr]
    call write_line
    lea rdx, [resolve_label]
    call write_line
    ret

emit_resolve:
    ; Emit mov/ret
    lea rdx, [resolve_instr]
    call write_line
    lea rdx, [newline]
    call write_line
    ; Emit end-of-scope marker
    lea rdx, [end_scope_marker]
    call write_line
    ; Simulate scope block end (if within a chamber)
    cmp qword [scope_depth], 0
    je .no_scope_exit
    dec qword [scope_depth]

    ; Pop from scope stack
    mov rbx, [scope_stack_ptr]
    cmp rbx, 0
    je .done
    dec rbx
    mov [scope_stack_ptr], rbx
.done:
.no_scope_exit:
    ret
.no_scope_exit:
    ret
    lea rdx, [resolve_instr]
    call write_line
    lea rdx, [newline]
    call write_line
    ; Simulate scope block end (if within a chamber)
    cmp qword [scope_depth], 0
    je .no_scope_exit
    dec qword [scope_depth]
.no_scope_exit:
    ret
    lea rdx, [resolve_instr]
    call write_line
    lea rdx, [newline]
    call write_line
    return/mov
    ret

emit_macro:
    ; Emit macro placeholder and simulate arg parsing
    lea rdx, [macro_instr]
    call write_line
    call emit_arguments
    ret

emit_rigid:
    ; Emit rigid block
    lea rdx, [rigid_instr]
    call write_line
    ret

write_line:
    ; Simulate writing instruction to console (stub for WriteFile)
    mov rcx, [console_handle]
    mov r8, strlen(rdx)
    lea r9, [bytes_written]
    sub rsp, 32
    call WriteConsoleA
    add rsp, 32
    ret

; Write label based on identifier_buffer
write_line_dynamic_label:
    ; Append colon to identifier and prefix with current scope
    lea rsi, [identifier_buffer]      ; user function name
    lea rdi, [token_buffer]           ; reuse token_buffer
    lea rbx, [scoped_prefix]          ; e.g., chamber_core_
    xor rcx, 0
    xor rdx, 0
.copy_scope:
    mov al, [rbx + rcx]
    cmp al, 0
    je .copy_name
    mov [rdi + rcx], al
    inc rcx
    jmp .copy_scope
.copy_name:
    mov al, [rsi + rdx]
    cmp al, 0
    je .version_hash
    mov [rdi + rcx], al
    inc rcx
    inc rdx
    jmp .copy_name
.version_hash:
    ; Append unique suffix _v{n}
    mov byte [rdi + rcx], '_'
    mov byte [rdi + rcx + 1], 'v'
    mov rax, [label_counter]
    add rax, '0'
    mov [rdi + rcx + 2], al
    add rax, 1
    mov [label_counter], rax
    mov byte [rdi + rcx + 3], ':'
    mov byte [rdi + rcx + 4], 0
    lea rdx, [token_buffer]
    call write_line
    ret

strcmp:
    xor eax, eax
.loop:
    mov al, [rdi]
    mov bl, [rsi]
    cmp al, bl
    jne .diff
    test al, al
    je .equal
    inc rdi
    inc rsi
    jmp .loop
.diff:
    mov eax, 1
    ret
.equal:
    xor eax, eax
    ret
.equal:
    xor eax, eax
    ret

section .data
    identifier_buffer db 128 dup(0)   ; Buffer for storing function/scope names
    token_buffer db 128 dup(0)        ; Temporary processing space
    scoped_prefix db "chamber_core_", 0  ; Example prefix
    label_counter dd 0                 ; Label version counter
    bytes_written dq 0                 ; Track console writes
    console_handle dq 0                 ; Simulated console handle
    new_line db 10, 0                   ; New line character for readability

section .text

; Simulates writing output to console, used for debugging label creation
write_line:
    mov rcx, [console_handle]       ; Console handle
    mov r8, strlen(rdx)             ; Determine message length
    lea r9, [bytes_written]         ; Track bytes written
    sub rsp, 32
    call WriteConsoleA
    add rsp, 32
    ret

; Function to construct a dynamic label from identifier + scope
write_line_dynamic_label:
    lea rsi, [identifier_buffer]  ; Function/variable name
    lea rdi, [token_buffer]       ; Label output buffer
    lea rbx, [scoped_prefix]      ; Scope prefix e.g., "chamber_core_"

    xor rcx, rcx  ; Initialize index
    xor rdx, rdx

.copy_scope:
    mov al, [rbx + rcx]
    cmp al, 0
    je .copy_name
    mov [rdi + rcx], al
    inc rcx
    jmp .copy_scope

.copy_name:
    mov al, [rsi + rdx]
    cmp al, 0
    je .version_hash
    mov [rdi + rcx], al
    inc rcx
    inc rdx
    jmp .copy_name

.version_hash:
    ; Append unique suffix `_v{n}`
    mov byte [rdi + rcx], '_'
    mov byte [rdi + rcx + 1], 'v'
    mov rax, [label_counter]
    add rax, '0'             ; Convert integer counter to ASCII
    mov [rdi + rcx + 2], al
    inc rax
    mov [label_counter], rax
    mov byte [rdi + rcx + 3], ':'   ; Label termination
    mov byte [rdi + rcx + 4], 0

    lea rdx, [token_buffer]
    call write_line                ; Output the generated label
    ret

; Basic string comparison utility
strcmp:
    xor eax, eax
.loop:
    mov al, [rdi]
    mov bl, [rsi]
    cmp al, bl
    jne .diff
    test al, al
    je .equal
    inc rdi
    inc rsi
    jmp .loop

.diff:
    mov eax, 1
    ret

.equal:
    xor eax, eax
    ret

section .data
    identifier_buffer db 128 dup(0)  ; Buffer for user-defined names
    token_buffer db 256 dup(0)       ; Expanded buffer to handle longer labels
    scoped_prefix db "chamber_", 0   ; Variable-length prefix
    label_counter dd 0               ; Unique label counter
    new_line db 10, 0                ; New line character
    max_label_size dd 255            ; Define max allowed label size

section .text

; Simulated console output helper
write_line:
    mov rcx, [console_handle]       ; Console handle
    mov r8, strlen(rdx)             ; Message length
    lea r9, [bytes_written]         ; Track bytes written
    sub rsp, 32
    call WriteConsoleA
    add rsp, 32
    ret

; Label generator with error handling
write_line_dynamic_label:
    lea rsi, [identifier_buffer]  ; Function name
    lea rdi, [token_buffer]       ; Label output buffer
    lea rbx, [scoped_prefix]      ; Scope prefix e.g., "chamber_"

    xor rcx, rcx  ; Reset index
    xor rdx, rdx

.check_overflow:
    mov eax, [max_label_size]    ; Max label length
    cmp rcx, eax
    ja .error_buffer_overflow

.copy_scope:
    mov al, [rbx + rcx]
    cmp al, 0
    je .copy_name
    mov [rdi + rcx], al
    inc rcx
    jmp .check_overflow

.copy_name:
    mov al, [rsi + rdx]
    cmp al, 0
    je .version_hash
    mov [rdi + rcx], al
    inc rcx
    inc rdx
    jmp .check_overflow

.version_hash:
    ; Append unique suffix "_v{n}"
    mov byte [rdi + rcx], '_'
    mov byte [rdi + rcx + 1], 'v'
    mov rax, [label_counter]
    add rax, '0'  ; Convert integer to ASCII
    mov [rdi + rcx + 2], al
    inc rax
    mov [label_counter], rax
    mov byte [rdi + rcx + 3], ':'  ; Label termination
    mov byte [rdi + rcx + 4], 0

    lea rdx, [token_buffer]
    call write_line                ; Output the generated label
    ret

.error_buffer_overflow:
    mov rdx, buffer_overflow_msg
    call write_line
    ret

section .data
buffer_overflow_msg db "ERROR: Label buffer overflow detected.", 0

section .data
    identifier_buffer db 128 dup(0)  ; Buffer for user-defined names
    token_buffer db 256 dup(0)       ; Expanded buffer for labels
    scoped_prefix db "chamber_", 0   ; Variable-length prefix
    label_counter dd 0               ; Unique label counter
    max_label_size dd 255            ; Define max allowed label size
    hash_table db 256 dup(0)         ; Hash table for duplicate detection
    error_msg db "ERROR: Label buffer overflow.", 0
    duplicate_msg db "ERROR: Duplicate label detected.", 0
    debug_output db "DEBUG: Generated label: ", 0
    new_line db 10, 0                ; New line character

section .text

; Simulated console output helper
write_line:
    mov rcx, [console_handle]
    mov r8, strlen(rdx)
    lea r9, [bytes_written]
    sub rsp, 32
    call WriteConsoleA
    add rsp, 32
    ret

; Hashing function for duplicate detection
hash_label:
    xor eax, eax
    xor ecx, ecx
.hash_loop:
    mov al, [rdi + ecx]
    test al, al
    je .store_hash
    add eax, ecx  ; Simple additive hash function
    inc ecx
    jmp .hash_loop
.store_hash:
    mov [hash_table + eax], 1
    ret

; Check for existing label in hash table
check_duplicate_label:
    xor eax, eax
    xor ecx, ecx
.check_loop:
    mov al, [rdi + ecx]
    test al, al
    je .verify_hash
    add eax, ecx
    inc ecx
    jmp .check_loop
.verify_hash:
    cmp byte [hash_table + eax], 1
    je .error_duplicate
    ret
.error_duplicate:
    mov rdx, duplicate_msg
    call write_line
    ret

; Label generator with error handling and namespace validation
write_line_dynamic_label:
    lea rsi, [identifier_buffer]  ; Function name
    lea rdi, [token_buffer]       ; Label output buffer
    lea rbx, [scoped_prefix]      ; Scope prefix

    xor rcx, rcx
    xor rdx, rdx

.check_overflow:
    cmp rcx, [max_label_size]
    ja .error_buffer_overflow

.copy_scope:
    mov al, [rbx + rcx]
    cmp al, 0
    je .copy_name
    mov [rdi + rcx], al
    inc rcx
    jmp .check_overflow

.copy_name:
    mov al, [rsi + rdx]
    cmp al, 0
    je .version_hash
    mov [rdi + rcx], al
    inc rcx
    inc rdx
    jmp .check_overflow

.version_hash:
    mov byte [rdi + rcx], '_'
    mov byte [rdi + rcx + 1], 'v'
    mov rax, [label_counter]
    add rax, '0'
    mov [rdi + rcx + 2], al
    inc rax
    mov [label_counter], rax
    mov byte [rdi + rcx + 3], ':'
    mov byte [rdi + rcx + 4], 0

.validate_namespace:
    call check_duplicate_label
    call hash_label

.debug_output:
    mov rdx, debug_output
    call write_line
    lea rdx, [token_buffer]
    call write_line

    ret

.error_buffer_overflow:
    mov rdx, error_msg
    call write_line
    ret

section .data
    identifier_buffer db 128 dup(0)  ; Buffer for user-defined names
    token_buffer db 256 dup(0)       ; Expanded buffer for labels
    scoped_prefix db "chamber_", 0   ; Variable-length prefix
    label_counter dd 0               ; Unique label counter
    max_label_size dd 255            ; Define max allowed label size
    hash_table db 256 dup(0)         ; Hash table for duplicate detection
    error_msg db "ERROR: Label buffer overflow.", 0
    duplicate_msg db "ERROR: Duplicate label detected.", 0
    debug_output db "DEBUG: Generated label: ", 0
    random_seed dq 123456789          ; Random seed for hash suffixing
    new_line db 10, 0                 ; New line character

section .text

; Simulated console output helper
write_line:
    mov rcx, [console_handle]
    mov r8, strlen(rdx)
    lea r9, [bytes_written]
    sub rsp, 32
    call WriteConsoleA
    add rsp, 32
    ret

; Hashing function for duplicate detection
hash_label:
    xor eax, eax
    xor ecx, ecx
.hash_loop:
    mov al, [rdi + ecx]
    test al, al
    je .store_hash
    add eax, ecx  ; Simple additive hash function
    inc ecx
    jmp .hash_loop
.store_hash:
    mov [hash_table + eax], 1
    ret

; Check for existing label in hash table
check_duplicate_label:
    xor eax, eax
    xor ecx, ecx
.check_loop:
    mov al, [rdi + ecx]
    test al, al
    je .verify_hash
    add eax, ecx
    inc ecx
    jmp .check_loop
.verify_hash:
    cmp byte [hash_table + eax], 1
    je .apply_random_suffix
    ret

.apply_random_suffix:
    mov rax, [random_seed]      ; Random seed
    imul rax, rax, 37           ; Random hash formula
    xor rax, 0xCAFEBABE         ; Mix with a fixed mask
    mov byte [rdi + rcx], '_'
    mov byte [rdi + rcx + 1], 'h'
    mov byte [rdi + rcx + 2], 'x'
    mov byte [rdi + rcx + 3], '0' + (rax mod 10)   ; Append randomized digit
    mov byte [rdi + rcx + 4], ':' 
    mov byte [rdi + rcx + 5], 0
    inc rax
    mov [random_seed], rax      ; Update seed to avoid repeats

.validate_namespace:
    call check_duplicate_label
    call hash_label

.debug_output:
    mov rdx, debug_output
    call write_line
    lea rdx, [token_buffer]
    call write_line

    ret

; Label generator with error handling and namespace validation
write_line_dynamic_label:
    lea rsi, [identifier_buffer]
    lea rdi, [token_buffer]
    lea rbx, [scoped_prefix]

    xor rcx, rcx
    xor rdx, rdx

.check_overflow:
    cmp rcx, [max_label_size]
    ja .error_buffer_overflow

.copy_scope:
    mov al, [rbx + rcx]
    cmp al, 0
    je .copy_name
    mov [rdi + rcx], al
    inc rcx
    jmp .check_overflow

.copy_name:
    mov al, [rsi + rdx]
    cmp al, 0
    je .version_hash
    mov [rdi + rcx], al
    inc rcx
    inc rdx
    jmp .check_overflow

.version_hash:
    mov byte [rdi + rcx], '_'
    mov byte [rdi + rcx + 1], 'v'
    mov rax, [label_counter]
    add rax, '0'
    mov [rdi + rcx + 2], al
    inc rax
    mov [label_counter], rax
    mov byte [rdi + rcx + 3], ':'
    mov byte [rdi + rcx + 4], 0

.validate_namespace:
    call check_duplicate_label
    call hash_label

.debug_output:
    mov rdx, debug_output
    call write_line
    lea rdx, [token_buffer]
    call write_line

    ret

.error_buffer_overflow:
    mov rdx, error_msg
    call write_line
    ret

section .data
    identifier_buffer db 128 dup(0)  ; Buffer for user-defined names
    token_buffer db 256 dup(0)       ; Expanded buffer for labels
    scoped_prefix db "chamber_", 0   ; Variable-length prefix
    label_counter dd 0               ; Unique label counter
    max_label_size dd 255            ; Max allowed label size
    hash_table db 256 dup(0)         ; Hash table for duplicate detection
    collision_cache db 256 dup(0)    ; Cache for resolved label conflicts
    error_msg db "ERROR: Label buffer overflow.", 0
    duplicate_msg db "ERROR: Duplicate label detected.", 0
    debug_output db "DEBUG: Generated label: ", 0
    random_seed dq 123456789          ; Random seed for hash suffixing
    mutex_lock db 0                   ; Mutex for thread synchronization
    new_line db 10, 0                 ; New line character

section .text

; Simulated console output helper
write_line:
    mov rcx, [console_handle]
    mov r8, strlen(rdx)
    lea r9, [bytes_written]
    sub rsp, 32
    call WriteConsoleA
    add rsp, 32
    ret

; Mutex lock for thread-safe execution
lock_mutex:
    mov al, [mutex_lock]
    test al, al
    jnz .wait
    mov byte [mutex_lock], 1
    ret
.wait:
    pause
    jmp lock_mutex

unlock_mutex:
    mov byte [mutex_lock], 0
    ret

; Hashing function for duplicate detection
hash_label:
    xor eax, eax
    xor ecx, ecx
.hash_loop:
    mov al, [rdi + ecx]
    test al, al
    je .store_hash
    add eax, ecx  ; Simple additive hash function
    inc ecx
    jmp .hash_loop
.store_hash:
    mov [hash_table + eax], 1
    ret

; Check for existing label in hash table
check_duplicate_label:
    xor eax, eax
    xor ecx, ecx
.check_loop:
    mov al, [rdi + ecx]
    test al, al
    je .verify_hash
    add eax, ecx
    inc ecx
    jmp .check_loop
.verify_hash:
    cmp byte [hash_table + eax], 1
    je .apply_random_suffix
    ret

; Collision resolution using randomized hash suffixing with caching
.apply_random_suffix:
    mov rax, [random_seed]
    imul rax, rax, 37
    xor rax, 0xCAFEBABE
    mov byte [rdi + rcx], '_'
    mov byte [rdi + rcx + 1], 'h'
    mov byte [rdi + rcx + 2], 'x'
    mov byte [rdi + rcx + 3], '0' + (rax mod 10)
    mov byte [rdi + rcx + 4], ':'
    mov byte [rdi + rcx + 5], 0
    mov [collision_cache + eax], 1  ; Store resolved conflict
    inc rax
    mov [random_seed], rax

.validate_namespace:
    call check_duplicate_label
    call hash_label

.debug_output:
    mov rdx, debug_output
    call write_line
    lea rdx, [token_buffer]
    call write_line

    ret

; Thread-safe label generator
write_line_dynamic_label:
    call lock_mutex

    lea rsi, [identifier_buffer]
    lea rdi, [token_buffer]
    lea rbx, [scoped_prefix]

    xor rcx, rcx
    xor rdx, rdx

.check_overflow:
    cmp rcx, [max_label_size]
    ja .error_buffer_overflow

.copy_scope:
    mov al, [rbx + rcx]
    cmp al, 0
    je .copy_name
    mov [rdi + rcx], al
    inc rcx
    jmp .check_overflow

.copy_name:
    mov al, [rsi + rdx]
    cmp al, 0
    je .version_hash
    mov [rdi + rcx], al
    inc rcx
    inc rdx
    jmp .check_overflow

.version_hash:
    mov byte [rdi + rcx], '_'
    mov byte [rdi + rcx + 1], 'v'
    mov rax, [label_counter]
    add rax, '0'
    mov [rdi + rcx + 2], al
    inc rax
    mov [label_counter], rax
    mov byte [rdi + rcx + 3], ':'
    mov byte [rdi + rcx + 4], 0

.validate_namespace:
    call check_duplicate_label
    call hash_label

.debug_output:
    mov rdx, debug_output
    call write_line
    lea rdx, [token_buffer]
    call write_line

    call unlock_mutex
    ret

.error_buffer_overflow:
    mov rdx, error_msg
    call write_line
    call unlock_mutex
    ret

section .text

copy_scope_simd:
    movdqu xmm0, [rbx]   ; Load 16 bytes from scoped prefix
    movdqu [rdi], xmm0   ; Store 16 bytes into token buffer
    add rdi, 16
    add rbx, 16
    ret

copy_name_simd:
    movdqu xmm1, [rsi]   ; Load 16 bytes from function identifier
    movdqu [rdi], xmm1   ; Store 16 bytes into token buffer
    add rdi, 16
    add rsi, 16
    ret

section .data
    start_cycle dq 0
    end_cycle dq 0
    perf_msg db "Elapsed CPU Cycles: ", 0

section .text

benchmark_label_gen:
    rdtsc                      ; Read starting timestamp
    mov [start_cycle], rdx
    call write_line_dynamic_label
    rdtsc                      ; Read ending timestamp
    mov [end_cycle], rdx
    sub rdx, [start_cycle]      ; Compute elapsed cycles

    mov rdx, perf_msg
    call write_line
    ret

