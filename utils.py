import torch


def push_and_pull(optimizer, local_net, global_net, terminal, next_state, buffer_state, buffer_action, buffer_reward,
                  gamma, device, isA3C=True):
    if terminal:
        next_state_value = 0.  # terminal
    else:
        next_state = torch.tensor(next_state, dtype=torch.float32).to(device)
        next_state_value = local_net.forward(next_state)[-1].cpu().detach().numpy()

    # 根据贝尔曼方程计算当前状态所能得到的回报总和
    buffer_v_target = []
    for reward in buffer_reward[::-1]:  # reverse buffer r
        next_state_value = reward + gamma * next_state_value
        buffer_v_target.append(next_state_value)
    buffer_v_target.reverse()
    # eps = np.finfo(np.float32).eps.item()
    # buffer_v_target = torch.tensor(buffer_v_target, dtype=torch.float32)
    # # 根据期望和方差做标准归一化
    # buffer_v_target = (buffer_v_target-buffer_v_target.mean())/(buffer_v_target.std()+eps)
    buffer_state = torch.stack(buffer_state)
    buffer_state = buffer_state.reshape(*buffer_state.shape[-4:])
    buffer_action = torch.stack(buffer_action).to(device)
    buffer_v_target = torch.tensor(buffer_v_target, dtype=torch.float32).reshape(-1, 1).to(device)
    loss = local_net.get_loss(buffer_state, buffer_action, buffer_v_target)

    if isA3C:
        # calculate local gradients and push local parameters to global
        optimizer.zero_grad()
        loss.backward()
        for lp, gp in zip(local_net.parameters(), global_net.parameters()):
            gp._grad = lp.grad
        optimizer.step()

        # pull global parameters
        local_net.load_state_dict(global_net.state_dict())
    else:
        return loss


def record(global_ep, global_ep_r, ep_r, res_queue, name):
    with global_ep.get_lock():
        global_ep.value += 1
    with global_ep_r.get_lock():
        if global_ep_r.value == 0.:
            global_ep_r.value = ep_r
        else:
            global_ep_r.value = global_ep_r.value * 0.99 + ep_r * 0.01
    res_queue.put(global_ep_r.value)
    print(name, "Ep:", global_ep.value, "| Ep_r: %.0f" % global_ep_r.value, )
