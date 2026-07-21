using FAIR.Domain.Entities.Chat;
namespace FAIR.Domain.Interfaces
{
    public interface IChatRepository
    {
        Task<Message> SaveMessageAsync(Message message);
        Task<List<Message>> GetPrivateMessagesAsync(string userId1, string userId2);
    }
}
