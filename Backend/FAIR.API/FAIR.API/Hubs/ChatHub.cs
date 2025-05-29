using FAIR.Application.DTOs.Chat;
using FAIR.Domain.Entities.Chat;
using FAIR.Domain.Interfaces;
using Microsoft.AspNetCore.SignalR;
using System.Collections.Concurrent;
namespace FAIR.API.Hubs
{
    public class ChatHub(IChatRepository chatRepository , IUserRepository userRepository ) : Hub
    {
        private readonly ConcurrentDictionary<string, OnlineUser> onlineUsers = new();
        private readonly IUserRepository _userRepository = userRepository;
        private readonly IChatRepository _chatRepository = chatRepository;
        public override async Task OnConnectedAsync()
        {
            var httpContext = Context.GetHttpContext();
            var recevierId = httpContext!.Request.Query["senderId"].ToString();
            var userName = Context.User!.Identity!.Name;
            var currentUser = await _userRepository.GetByUsernameAsync(userName!);
            var connectionId = Context.ConnectionId;
            if (onlineUsers.ContainsKey(connectionId))
            {
                onlineUsers[userName!].ConnectionId = connectionId;
            }
            else
            {
                var user = new OnlineUser
                {
                    UserName = userName,
                    ConnectionId = connectionId,
                    FullName = currentUser.UserName
                };
                onlineUsers.TryAdd(connectionId, user);
            }
        }
        public async Task SendMessage(MessageRequest message)
        {
            var senderId = Context.User!.Identity!.Name;
            var receiverId = message.ReceiverId;

            var newMess = new Message
            {
                Sender = await _userRepository.GetByIdAsync(senderId!),
                ReceiverId = receiverId,
                IsRead = false,
                Content = message.Content,
                CreateData = DateTime.Now

            };
            await _chatRepository.SaveMessageAsync(newMess);
            await Clients.User(receiverId!).SendAsync("RecivedMessage", newMess);

        }

    }
}
